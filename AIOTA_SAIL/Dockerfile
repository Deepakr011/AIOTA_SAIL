# Use strict versioning for production stability
FROM python:3.10-slim-bookworm

# Set working directory inside the container
WORKDIR /app

# 1. Install system tools required for building Python extensions
# (curl/healthcheck are useful for debugging later)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 2. Copy dependencies first (Optimization: Docker caches this layer)
COPY requirements.txt .

# 3. Install Python libraries
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy your source code and data files
COPY . .

# 5. Run the script with "unbuffered" output so you see logs instantly
CMD ["python", "-u", "mqtt_llm.py"]