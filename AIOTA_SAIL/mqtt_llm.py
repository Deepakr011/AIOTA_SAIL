import json
import os
import re
import time
import sqlite3
import redis
import threading
import paho.mqtt.client as mqtt
from openai import OpenAI
import chromadb
from datetime import datetime

# ================= 1. CONFIGURATION =================
MQTT_BROKER = "mosquitto"
REDIS_HOST = "redis"
CHAT_URL = "http://llama_chat:8080/v1"
EMBED_URL = "http://llama_embed:8081/v1"

TOPIC_QUERY = "aiota/query"
TOPIC_STREAM = "aiota/stream/"

SESSION_TTL = 1200      # 20 Minutes
MAX_HISTORY_DEPTH = 6   
DB_FILE = "/app/data/chat_logs.db"

CACHE_MASTER_CONTEXT = {} 


# ================= 2. CONNECTIONS =================
while True:
    try:
        r_cache = redis.Redis(host=REDIS_HOST, port=6379, db=0)
        r_cache.ping()
        print("Redis Connected")
        break
    except redis.ConnectionError:
        print("Waiting for Redis...")
        time.sleep(2)

client_chat = OpenAI(base_url=CHAT_URL, api_key="sk-no-key")
client_embed = OpenAI(base_url=EMBED_URL, api_key="sk-no-key")

chroma_client = chromadb.PersistentClient(path="/app/chroma_db")
collection = chroma_client.get_collection("rag_chunks")

os.makedirs("/app/data", exist_ok=True)
conn = sqlite3.connect(DB_FILE)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS logs
             (timestamp TEXT, display_id INT, role TEXT, content TEXT, experiment INT, intent TEXT)''')
conn.commit()
conn.close()

# ================= 3. LOGIC HELPER FUNCTIONS =================

def get_embedding(text):
    try:
        res = client_embed.embeddings.create(model="nomic", input=text)
        return res.data[0].embedding
    except Exception as e:
        print(f"❌ Embed Error: {e}")
        return []

def detect_intent(query: str) -> str:
    q = query.lower()
    if any(k in q for k in ["explain", "overview", "intro", "about"]): return "overview"
    if any(k in q for k in ["how", "connect", "wiring", "pinout"]): return "how_to"
    if any(k in q for k in ["why", "reason", "concept"]): return "theory"
    if any(k in q for k in ["output", "observe", "result"]): return "expected_output"
    if any(k in q for k in ["error", "fail", "not working", "stuck"]): return "debug"
    return "general"

def detect_experiment(query: str, current_exp):
    if not query: return None
    q = query.strip().lower()

    if q.isdigit():
        return int(q)

    match = re.search(r"(experiment|exp|esp|lab)\s*(no\.?)?\s*(\d+)", q)
    if match: return int(match.group(3))
    
    return None
def get_master_context(experiment_id):
    """
    Robust Context Fetcher:
    1. Tries specific tags ('experiment_context', 'master').
    2. If missing, runs a vector search for 'summary' to find the best intro.
    """
    if experiment_id in CACHE_MASTER_CONTEXT:
        return CACHE_MASTER_CONTEXT[experiment_id]
    
    print(f"🔎 Searching DB for Exp {experiment_id}...")

    # A. ATTEMPT STRICT TAGS
    for tag in ["experiment_context", "master", "intro"]:
        # Try Integer
        results = collection.get(
            where={"$and": [{"experiment": experiment_id}, {"chunk_type": tag}]},
            limit=1
        )
        # Try String
        if not results["documents"]:
            results = collection.get(
                where={"$and": [{"experiment": str(experiment_id)}, {"chunk_type": tag}]},
                limit=1
            )

        if results["documents"]:
            print(f"✅ Found context via tag: '{tag}'")
            text = results["documents"][0]
            CACHE_MASTER_CONTEXT[experiment_id] = text
            return text

    # B. VECTOR FALLBACK (Self-Healing)
    emb = get_embedding(f"Summary of Experiment {experiment_id} goal and components")
    if not emb: return None

    results = collection.query(
        query_embeddings=[emb],
        n_results=1,
        where={"experiment": experiment_id}
    )
    
    if not results or not results.get("documents") or len(results["documents"][0]) == 0:
        results = collection.query(
            query_embeddings=[emb],
            n_results=1,
            where={"experiment": str(experiment_id)}
        )

    if results and results.get("documents") and len(results["documents"][0]) > 0:
        text = results["documents"][0][0]
        CACHE_MASTER_CONTEXT[experiment_id] = text
        return text
    return None

def retrieve_rag_chunks(query, experiment):
    emb = get_embedding(query)
    if not emb: return []
    
    # 1. Experiment Specific Chunks
    try:
        exp_results = collection.query(
            query_embeddings=[emb],
            n_results=3,
            where={"$and": [{"experiment": experiment}, {"chunk_type": {"$ne": "experiment_context"}}]}
        )
        exp_text = exp_results["documents"][0] if exp_results.get("documents") else []
    except:
        exp_text = []

    # 2. Device Specs - FOCUSED SEARCH
    context_topic = f"Experiment {experiment} hardware components"
    focused_query = f"{query} {context_topic}"
    dev_emb = get_embedding(focused_query)

    dev_results = collection.query(
        query_embeddings=[dev_emb],
        n_results=2,
        where={"source": "device"}
    )
    
    # Fallback to content_type if source fails
    if not dev_results.get("documents") or len(dev_results["documents"][0]) == 0:
         dev_results = collection.query(
            query_embeddings=[dev_emb],
            n_results=2,
            where={"content_type": "device_info"}
        )
    
    dev_text = dev_results["documents"][0] if dev_results.get("documents") else []

    return exp_text + dev_text

# ================= 4. SESSION & STATE =================

def get_session_state(display_id):
    key = f"state:{display_id}"
    exp = r_cache.get(key)
    return int(exp) if exp else None

def set_session_state(display_id, exp_id):
    key = f"state:{display_id}"
    r_cache.set(key, exp_id, ex=SESSION_TTL)

def clear_session(display_id):
    r_cache.delete(f"session:{display_id}")
    r_cache.delete(f"state:{display_id}")

def get_context_window(display_id):
    items = r_cache.lrange(f"session:{display_id}", 0, -1)
    history = ""
    for item in items:
        msg = json.loads(item)
        role = "Student" if msg["role"] == "user" else "AIOTA"
        history += f"{role}: {msg['content']}\n"
    return history

def save_turn(display_id, user_text, ai_text, experiment_id, intent="general"):
    redis_key = f"session:{display_id}"
    r_cache.rpush(redis_key, json.dumps({"role": "user", "content": user_text}))
    r_cache.rpush(redis_key, json.dumps({"role": "assistant", "content": ai_text}))
    if r_cache.llen(redis_key) > MAX_HISTORY_DEPTH:
        r_cache.lpop(redis_key, r_cache.llen(redis_key) - MAX_HISTORY_DEPTH)
    
    r_cache.expire(redis_key, SESSION_TTL)
    r_cache.expire(f"state:{display_id}", SESSION_TTL)

    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        ts = datetime.now().isoformat()
        c.execute("INSERT INTO logs VALUES (?,?,?,?,?,?)", 
                 (ts, display_id, "user", user_text, experiment_id, intent))
        c.execute("INSERT INTO logs VALUES (?,?,?,?,?,?)", 
                 (ts, display_id, "assistant", ai_text, experiment_id, intent))
        conn.commit()
        conn.close()
    except: pass

# ================= 5. CORE PROCESSOR =================

def process_question(display_id, question):
    print(f"⚡ Processing ID {display_id}...")
    topic_out = f"{TOPIC_STREAM}{display_id}"
    q_clean = question.strip().lower()

    # --- 1. HANDLE RESET ---
    if q_clean in ["reset", "/reset", "clear", "restart"]:
        clear_session(display_id)
        mqtt_client.publish(topic_out, "Session Reset. Please enter the Experiment Number. <END>")
        return

    # --- 2. DETECT EXPERIMENT SWITCH ---
    current_exp = get_session_state(display_id)
    detected_exp = detect_experiment(question, current_exp)
    
    if isinstance(detected_exp, int):
        clear_session(display_id)
        set_session_state(display_id, detected_exp)
        
        master_context = get_master_context(detected_exp)
        if not master_context:
            mqtt_client.publish(topic_out, f"Switched to Exp {detected_exp}. (Manual unavailable). <END>")
            return

        # --- TEXT-ONLY WELCOME PROMPT ---
        welcome_prompt = f"""
You are IOTA SAIL, an on-device IoT lab assistant. 

Create a welcome message for **Experiment {detected_exp}** using ONLY the text below.
Start EXACTLY with: "Welcome to Experiment {detected_exp}: <Title from text>".
In one sentence, state the Aim. In one short question, ask the student if they are ready to begin.
Use only information from the text, add no general theory.

TEXT:
{master_context}
"""

        try:
            full_wel = ""
            stream = client_chat.chat.completions.create(
                model="default",
                messages=[{"role": "system", "content": welcome_prompt}],
                max_tokens=150, temperature=0.1, stream=True 
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    text = chunk.choices[0].delta.content.replace("*", "").replace("- ", "")
                    mqtt_client.publish(topic_out, text)
                    full_wel += text
            mqtt_client.publish(topic_out, "<END>")
            save_turn(display_id, f"Start Exp {detected_exp}", full_wel, detected_exp, "start")
            return
        except:
            mqtt_client.publish(topic_out, f"Exp {detected_exp} Ready. <END>")
            return

    # --- 3. CHECK SESSION ---
    if current_exp is None:
        mqtt_client.publish(
            topic_out,
            "I can assist only within an active experiment. "
            "Please specify the experiment number (e.g., Experiment 2 or lab 2) to continue. <END>"
            )
        return

    # --- 4. Q&A GENERATION ---
    target_exp = current_exp
    intent = detect_intent(question)
    
    master_context = get_master_context(target_exp) or "Check manual."
    chunks = retrieve_rag_chunks(question, target_exp)
    chunk_text = ' '.join(chunks) if chunks else "No specific steps found. Refer to context."
    history = get_context_window(display_id)

    # --- TEXT-ONLY Q&A PROMPT ---
    system_prompt = f"""
You are IOTA SAIL, an on-device IoT lab assistant. 
Experiment: {target_exp}.
Intent: {intent.upper()}

STRICT RULES:
- Answer ONLY using the provided experiment context.
- If the answer is not present, ask the student to refer to the experiment manual or instructor.
- Do NOT introduce general electronics theory.
- Keep the answer within 3–5 sentences.
- Be practical and lab-focused student-friendly.

CONTEXT:
{master_context}

DETAILS:
{chunk_text}
"""
    user_prompt = f"HISTORY:\n{history}\nQUESTION: {question}"

    full_answer = ""
    try:
        stream = client_chat.chat.completions.create(
            model="default",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=300, temperature=0.1, stream=True
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                text = chunk.choices[0].delta.content.replace("*", "").replace("- ", "")
                mqtt_client.publish(topic_out, text)
                full_answer += text
        
        mqtt_client.publish(topic_out, "<END>")
        save_turn(display_id, question, full_answer, target_exp, intent)
        print(f"✅ Finished ID {display_id}")
    except Exception as e:
        print(f"❌ Error: {e}")
        mqtt_client.publish(topic_out, "System Error. <END>")

# ================= 6. PARALLEL WORKER =================
def on_mqtt_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        r_cache.rpush("job_queue", json.dumps(payload))
    except: pass

def worker_thread(thread_id):
    print(f"🔧 Worker {thread_id} ready.")
    while True:
        try:
            res = r_cache.blpop("job_queue", timeout=2)
            if res:
                process_question(json.loads(res[1]).get("id"), json.loads(res[1]).get("text"))
        except: pass

mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
mqtt_client.connect(MQTT_BROKER, 1883, 60)
mqtt_client.subscribe(TOPIC_QUERY)
mqtt_client.on_message = on_mqtt_message
mqtt_client.loop_start()

print("🚀 System Ready. Launching 2 Parallel Workers...")
threads = []
for i in range(2): 
    t = threading.Thread(target=worker_thread, args=(i+1,))
    t.daemon = True 
    t.start()
    threads.append(t)

try:
    while True: time.sleep(1)
except KeyboardInterrupt:
    mqtt_client.loop_stop()