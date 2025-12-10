# FastAPI Backend for Token Entropy Analysis

Production-ready REST API for analyzing LLM token entropy and detecting risky code patterns.

## 🚀 Quick Start with Docker

### Option 1: Using Docker Compose (Recommended)

From the **root directory** of the project:

```bash
# Start the API server
docker-compose up api

# Or run in background
docker-compose up -d api
```

The API will be available at: **http://localhost:8000**

### Option 2: Docker Build Manually

From the **backend directory**:

```bash
# Build the image
docker build -t logit-sec-probe-api .

# Run the container
docker run -d \
  -p 8000:8000 \
  -v logit-sec-probe-hf-cache:/app/.cache/huggingface \
  --name logit-sec-probe-api \
  logit-sec-probe-api
```

## 📡 API Endpoints

### GET `/`
Health check endpoint.

```bash
curl http://localhost:8000/
```

**Response:**
```json
{
  "status": "ok",
  "message": "Token Entropy Analysis API is running",
  "model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
  "device": "cuda:0"
}
```

### GET `/health`
Detailed health status.

```bash
curl http://localhost:8000/health
```

### POST `/analyze`
Analyze token-level entropy for code generation.

**Request Body:**
```json
{
  "prompt": "Write a C function to copy a user-provided string into a fixed-size buffer.",
  "system_prompt": "You are a secure coding assistant.",
  "max_tokens": 100,
  "risky_keywords": ["strcpy", "gets", "system", "eval"]
}
```

**Example cURL:**
```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a C function to copy strings",
    "system_prompt": "You are a secure coding assistant.",
    "max_tokens": 100,
    "risky_keywords": ["strcpy", "gets"]
  }'
```

**Response:**
```json
{
  "generated_text": "Here's a safe string copy function...",
  "tokens": [
    {
      "token": "Here",
      "entropy": 0.123456,
      "prob": 0.950000,
      "is_risky": false
    },
    {
      "token": "strcpy",
      "entropy": 0.678901,
      "prob": 0.420000,
      "is_risky": true
    }
  ],
  "metadata": {
    "total_tokens": 50,
    "risky_tokens_count": 1,
    "avg_entropy": 0.234567,
    "avg_probability": 0.876543,
    "max_entropy": 0.901234,
    "min_probability": 0.123456
  }
}
```

## 🔧 Local Development (without Docker)

If you prefer to run locally:

```bash
cd backend

# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## 🐳 Docker Commands

### Start the API
```bash
docker-compose up api
```

### Stop the API
```bash
docker-compose down
```

### View logs
```bash
docker-compose logs -f api
```

### Rebuild after code changes
```bash
docker-compose build api
docker-compose up api
```

### Run research script (batch analysis)
```bash
docker-compose --profile research up entropy-analysis
```

## 🎯 GPU Support

To enable GPU acceleration, uncomment the GPU section in `docker-compose.yml`:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

**Requirements:**
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- Compatible NVIDIA GPU

## 📊 Interactive API Documentation

FastAPI automatically generates interactive documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🔐 Security Notes

- CORS is configured to allow all origins (`allow_origins=["*"]`)
- For production, restrict CORS to specific domains
- Consider adding authentication middleware for production deployments

## 🧪 Testing the API

### Python Example
```python
import requests

response = requests.post(
    "http://localhost:8000/analyze",
    json={
        "prompt": "Write a buffer overflow example",
        "risky_keywords": ["strcpy", "gets"]
    }
)

data = response.json()
print(f"Generated: {data['generated_text']}")
print(f"Risky tokens: {data['metadata']['risky_tokens_count']}")
```

### JavaScript Example
```javascript
fetch('http://localhost:8000/analyze', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    prompt: 'Write a SQL query builder',
    risky_keywords: ['execute', 'eval']
  })
})
.then(r => r.json())
.then(data => console.log(data));
```

## 📦 Model Caching

The model (~3GB) is cached in a Docker volume `hf-cache` to avoid re-downloading on every container restart. This volume is shared between the API and research services.

## 🛠️ Troubleshooting

**Model not loading:**
- Check logs: `docker-compose logs api`
- Ensure sufficient disk space (~5GB)
- First run takes 3-5 minutes to download model

**Port already in use:**
```bash
# Change port in docker-compose.yml
ports:
  - "8001:8000"  # Use port 8001 instead
```

**Out of memory:**
- The model requires ~4GB RAM minimum
- Consider using a smaller model or adding swap space
