import paho.mqtt.client as mqtt
import json
import time
import threading
import random

# ================= CONFIGURATION =================
MQTT_BROKER = "localhost"  # Change to your Docker IP if running externally
MQTT_PORT = 1883
TOPIC_QUERY = "aiota/query"
TOPIC_STREAM_WILDCARD = "aiota/stream/#"

# Define the 9 Simulated Students (ID, Experiment, Question)
STUDENTS = [
    {"id": 1, "exp": "1", "q": "What is the GPIO pin for the LED?"},
    {"id": 2, "exp": "5", "q": "Explain the PWM duty cycle logic."},
    {"id": 3, "exp": "2", "q": "How do I wire the push button?"},
    {"id": 4, "exp": "3", "q": "Why do we need a debounce delay?"},
    {"id": 5, "exp": "4", "q": "What is the ADC resolution used here?"},
    {"id": 6, "exp": "1", "q": "Explain the code loop function."},
    {"id": 7, "exp": "5", "q": "What happens if I connect the motor backwards?"},
    {"id": 8, "exp": "2", "q": "What is the expected output on serial monitor?"},
    {"id": 9, "exp": "3", "q": "Explain the edge detection logic."}
]

# ================= MQTT SETUP =================
def on_connect(client, userdata, flags, rc, properties=None): # Updated for V2
    if rc == 0:
        print("✅ Load Tester Connected to MQTT")
        client.subscribe(TOPIC_STREAM_WILDCARD)
    else:
        print(f"❌ Connection Failed: {rc}")

def on_message(client, userdata, msg):
    """
    Listens for responses to verify the server is working.
    Prints a short snippet of the answer.
    """
    try:
        # Extract Display ID from topic "aiota/stream/5" -> "5"
        display_id = msg.topic.split("/")[-1]
        payload = msg.payload.decode()
        
        # Don't print <END> tags to keep console clean
        if "<END>" not in payload:
            print(f"   📩 [Display {display_id} Response]: {payload[:60]}...")
        elif "Welcome" in payload:
            print(f"   🎉 [Display {display_id}]: Received WELCOME Message")
    except Exception as e:
        print(f"Error parsing message: {e}")

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_message = on_message
client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.loop_start()

# ================= SIMULATION WORKER =================
def simulate_student(student_data):
    did = student_data["id"]
    exp = student_data["exp"]
    question = student_data["q"]

    # --- STEP 1: RESET SESSION ---
    # Mimics a student sitting down and clearing previous data
    print(f"🚀 [Display {did}] Sending: RESET")
    msg_reset = json.dumps({"id": did, "text": "reset"})
    client.publish(TOPIC_QUERY, msg_reset)
    
    # Wait 2 seconds (Realism + Processing time)
    time.sleep(2)

    # --- STEP 2: SELECT EXPERIMENT (Triggers Welcome) ---
    # This is a "Heavy" LLM task
    print(f"🧪 [Display {did}] Selecting: Experiment {exp}")
    msg_exp = json.dumps({"id": did, "text": str(exp)})
    client.publish(TOPIC_QUERY, msg_exp)

    # Wait 8-12 seconds (Simulate reading the welcome message + Queue wait)
    # Since we have 2 parallel workers, some requests will queue up.
    delay = random.uniform(8, 12) 
    time.sleep(delay)

    # --- STEP 3: ASK QUESTION (Triggers RAG) ---
    # This is a "Heavy" RAG + LLM task
    print(f"❓ [Display {did}] Asking: '{question}'")
    msg_q = json.dumps({"id": did, "text": question})
    client.publish(TOPIC_QUERY, msg_q)

# ================= EXECUTION =================
if __name__ == "__main__":
    print(f"🔥 Starting Load Test with {len(STUDENTS)} Parallel Students...")
    print("---------------------------------------------------------------")
    
    threads = []
    
    # Launch all students at once (Parallel burst)
    for student in STUDENTS:
        t = threading.Thread(target=simulate_student, args=(student,))
        t.start()
        threads.append(t)
        # Stagger slightly (0.5s) so they don't hit MQTT exact nanosecond
        time.sleep(0.5) 

    # Wait for all threads to finish sending
    for t in threads:
        t.join()

    print("---------------------------------------------------------------")
    print("✅ All prompts sent. Watching for streaming responses...")
    print("   (Press Ctrl+C to stop monitoring)")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Test Finished.")
        client.loop_stop()