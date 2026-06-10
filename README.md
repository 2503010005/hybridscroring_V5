# HybridScoring_V5

AI-Assisted Decision Support System for Indonesian Essay Assessment using a Context-Aware Hybrid CNN–BiLSTM Model.

## Overview

HybridScoring_V5 is an educational decision support system designed to assist lecturers in evaluating Indonesian essays. The system integrates a trained **Context-Aware Hybrid CNN–BiLSTM** model with a web-based interface and REST API deployment to provide:

* AI-generated score recommendations,
* automatic feedback generation,
* lecturer validation mechanisms,
* dynamic rubric integration, and
* human-in-the-loop assessment workflows.

The system is intended to support academic assessment processes rather than replace human evaluators.

---

## Research Background

This repository represents the implementation stage of a broader research roadmap consisting of:

1. **Conference Paper**

   * Hybrid AI scoring using semantic similarity, rubric-based scoring, and generative approaches.

2. **IJCCS Journal**

   * Development of a Context-Aware Hybrid CNN–BiLSTM model enhanced with semantic feature injection and generative augmentation.

3. **Educational DSS Implementation**

   * Integration of the trained model into a deployable AI-assisted decision support system accessible through API and web interfaces.

---

## Features

### Automated Essay Scoring

* Context-aware essay representation.
* CNN-based local linguistic feature extraction.
* BiLSTM-based sequential context modeling.
* Semantic feature injection using Indonesian SBERT embeddings.
* Hybrid generative augmentation for training enhancement.

### Educational Decision Support

* Instructor-oriented assessment interface.
* Human-in-the-loop validation.
* AI score recommendation.
* Lecturer score comparison.
* Feedback generation support.

### Deployment

* REST API implementation.
* Web-based demonstration application.
* Modular architecture for easier maintenance and debugging.

---

## System Architecture

```text
Student Essay
      │
      ▼
Preprocessing
      │
      ▼
Semantic Feature Injection
      │
      ▼
Context-Aware Hybrid CNN–BiLSTM
      │
      ▼
Predicted Score
      │
      ├────────► AI Feedback
      │
      ▼
REST API
      │
      ▼
Instructor Workspace
      │
      ▼
Lecturer Validation
      │
      ▼
Final Assessment Decision
```

---

## Project Structure

```text
HybridScoring_V5/
│
├── api/                    # FastAPI backend
├── client_demo/            # Frontend application
├── models/                 # Trained models
├── preprocessing/          # Data preparation utilities
├── training/               # Model training scripts
├── feedback/               # Feedback generation modules
├── config/                 # Configuration files
├── notebooks/              # Experimental notebooks
├── dataset/                # Dataset references
├── requirements.txt
└── README.md
```

---

## Dataset

The study utilized **4,500 Indonesian essay responses** scored by lecturers as ground truth.

Dataset split:

| Dataset  | Size  |
| -------- | ----- |
| Training | 3,600 |
| Testing  | 900   |

The dataset captures variations in:

* conceptual understanding,
* response quality,
* writing characteristics, and
* academic contexts.

---

## Model Performance

Performance was evaluated using standard Automated Essay Scoring metrics.

| Metric                         | Score |
| ------------------------------ | ----- |
| Quadratic Weighted Kappa (QWK) | 0.888 |
| R² Score                       | 0.816 |

These results indicate substantial agreement between model recommendations and lecturer evaluations.

---

## API Endpoints

### Health Check

```http
GET /v1/health
```

---

### Predict Essay Score

```http
POST /v1/predict
```

Request body:

```json
{
    "student_answer": "...",
    "reference_answer": "...",
    "course": "Artificial Intelligence"
}
```

Response example:

```json
{
    "score": 84.5,
    "model_metrics": {
        "qwk": 0.888,
        "r2": 0.816
    }
}
```

---

### Get Available Courses

```http
GET /v1/courses
```

---

## Installation

Clone repository:

```bash
git clone https://github.com/2503010005/hybridscroring_V5.git

cd hybridscroring_V5
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the API

```bash
python api/main.py
```

Default:

```text
http://localhost:8888
```

---

## Running the Demo Application

```bash
streamlit run client_demo/app_demo.py
```

---

## Educational Use

The system is designed as an **AI-assisted educational decision support system**, meaning that:

* AI provides recommendations,
* lecturers retain full authority over final grading decisions,
* assessment accountability remains with human evaluators.

The implementation follows a **human-in-the-loop** paradigm to support responsible AI adoption in education.

---

## Publications

Related publications associated with this repository include:

1. Hybrid AI-based Essay Scoring Conference Paper.
2. Context-Aware Hybrid CNN–BiLSTM Journal Article.
3. AI-Assisted Educational Decision Support System Deployment Study.

---

## Citation

If you use this repository in academic work, please cite the corresponding publications.

```bibtex
@article{hybridscoring2026,
  title={AI-Assisted Decision Support System for Indonesian Essay Assessment Using a Context-Aware Hybrid CNN--BiLSTM Approach},
  author={Sri Artha, I. M. G.},
  year={2026}
}
```

---

## License

This project is intended for educational and research purposes.

Please contact the author for collaboration or permission regarding commercial applications.
