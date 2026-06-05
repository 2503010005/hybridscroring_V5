1. cd /var/apps/eas_api
2. python -m venv venv_eas
3. source venv_eas/bin/activate
4. pip install -r requirements.txt
5. uvicorn app:app --reload --port 8001
6. API Dokumentasi /docs


# Revisi kecil `app.py` (WAJIB)

Tambahkan:

```python
API_PREFIX = "/v1"
```

dan ubah endpoint.

### BEFORE

```python
@app.get("/")
@app.get("/courses")
@app.post("/predict")
@app.get("/health")
```

### AFTER

```python
API_PREFIX = "/v1"


@app.get(f"{API_PREFIX}/")
@app.get(f"{API_PREFIX}/courses")
@app.post(f"{API_PREFIX}/predict")
@app.get(f"{API_PREFIX}/health")
```

Final endpoint:

| Endpoint      | Method |
| ------------- | ------ |
| `/v1/`        | GET    |
| `/v1/courses` | GET    |
| `/v1/predict` | POST   |
| `/v1/health`  | GET    |

---

# `README_API.md`

Letakkan:

```text
api/README_API.md
```

Isi berikut.

---

# Hybrid AES API v4500

Context-Aware Hybrid Spatio-Sequential Automated Essay Scoring API for Indonesian Essay Assessment.

The API implements a CNN–BiLSTM Hybrid AES model with:

* TF-IDF similarity
* SBERT semantic feature injection
* Course-aware metadata
* Hybrid spatio-sequential learning
* Context-aware scoring

The deployed model corresponds to:

```text
hybrid_model_v4500.keras
```

---

# Project Structure

```text
hybridscoring_V5/
│
├── config.py
├── main.py
│
├── model_v5/
│   ├── hybrid_model_v4500.keras
│   ├── tokenizer_v4500.pkl
│   ├── scaler_v4500.pkl
│   ├── course_encoder_v4500.pkl
│   ├── proxy_model_v4500.pkl
│   ├── evaluation_v4500.json
│   └── training_history_v4500.csv
│
├── api/
│   ├── app.py
│   ├── README_API.md
│   └── utils/
```

---

# Installation

Activate virtual environment:

```bash
source venv10/bin/activate
```

Install API dependencies if not yet available:

```bash
pip install fastapi uvicorn pydantic
```

Training dependencies remain unchanged because the API reuses:

```text
requirements.txt
```

---

# Run API

From project root:

```bash
cd api
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Successful startup:

```text
Application startup complete.
```

Swagger UI:

```text
http://localhost:8000/docs
```

OpenAPI:

```text
http://localhost:8000/openapi.json
```

---

# API Endpoints

## 1. Status + Model Metrics

Endpoint:

```text
GET /v1/
```

Example:

```bash
curl http://localhost:8000/v1/
```

Response:

```json
{
  "status":"ok",
  "model":"hybrid_model_v4500",
  "metrics":{
    "qwk":0.888,
    "r2":0.816,
    "rmse":15.73,
    "mae":12.43,
    "mse":247.58
  },
  "numeric_dim":772,
  "max_sequence_length":250
}
```

---

## 2. Available Courses

Endpoint:

```text
GET /v1/courses
```

Example:

```bash
curl http://localhost:8000/v1/courses
```

Response:

```json
{
  "total_courses":4,
  "courses":[
    "Artificial Intelligence",
    "Database",
    "Computer Network",
    "Professional Ethics"
  ]
}
```

Only trained courses are supported.

---

## 3. Essay Prediction

Endpoint:

```text
POST /v1/predict
```

Request:

```json
{
  "student_answer":"Artificial intelligence ...",
  "reference_answer":"AI is a branch ...",
  "course":"Artificial Intelligence"
}
```

Example curl:

```bash
curl -X POST \
http://localhost:8000/v1/predict \
-H "Content-Type: application/json" \
-d '{
    "student_answer":"...",
    "reference_answer":"...",
    "course":"Artificial Intelligence"
}'
```

Response:

```json
{
  "score":84.72,
  "normalized_score":0.8472,
  "course":"Artificial Intelligence",
  "model_metrics":{
      "qwk":0.888,
      "r2":0.816
  }
}
```

---

## 4. Health Check

Endpoint:

```text
GET /v1/health
```

Response:

```json
{
  "status":"healthy"
}
```

---

# Strict Course Validation

The API applies strict course validation.

If a course was not used during training:

Request:

```json
{
  "course":"Operating System"
}
```

Response:

```json
{
  "detail":{
      "error":"course not trained",
      "available_courses":[...]
  }
}
```

HTTP:

```text
400 Bad Request
```

Retraining is recommended for new academic domains.

---

# Inference Pipeline

The deployed inference pipeline strictly follows the training architecture.

```text
Student Answer
+
Reference Answer
+
Course
↓
Tokenizer
↓
TF-IDF Similarity
↓
SBERT Embedding
↓
SBERT Similarity
↓
Length Feature
↓
Course Encoder
↓
Feature Scaling
↓
Hybrid CNN-BiLSTM
↓
Essay Score
```

Numeric feature dimension:

```text
772
```

This ensures consistent inference without feature mismatch.

