# Emotion-to-Action AI Pipeline

An end-to-end ML pipeline that transforms reflective journal entries into actionable mental wellness recommendations.

## 🚀 Features

- **Multi-Modal Emotion Classification**: Predicts 6 emotional states using calibrated `LinearSVC`.
- **Intensity Mapping**: 5-level intensity prediction via `GradientBoostingClassifier`.
- **Advanced NLP**: Word & Character n-gram TF-IDF + structured feature interactions.
- **Supportive Conversational System**: Generates human-like, contextualized responses based on emotional state.
- **Uncertainty-Aware**: Detects and hedges responses for low-confidence predictions.
- **FastAPI Endpoint**: Ready-for-deployment `POST /predict` API.
- **Streamlit UI**: Premium dark glassmorphism dashboard for real-time analysis.

## 📂 Project Structure

```text
ml-emotion-assistant/
├── app.py                 # Streamlit UI Demo
├── main.py                # Full batch pipeline
├── src/
│   ├── api/               # FastAPI implementation
│   ├── decision_engine/   # Rules, Schedulers & Message Templates
│   ├── evaluation/        # Metrics & Label Noise handling
│   ├── inference/         # Model loading & Uncertainty estimation
│   ├── models/            # Training logic
│   ├── preprocessing/     # Text cleaning & Feature engineering
│   └── utils/             # Config & Helpers
├── reports/               # Error analysis & documentation
├── outputs/               # Saved models & predictions
└── data/                  # Raw & Processed datasets
```

## 🛠️ Setup & Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Batch Inference
```bash
python main.py
```
Outputs are saved to `outputs/predictions.csv` and `outputs/low_confidence_log.csv`.

### 3. Launch the API (FastAPI)
```bash
uvicorn src.api.app:app --port 8000
```
Interactive docs available at `http://localhost:8000/docs`.

### 4. Launch the UI (Streamlit)
```bash
streamlit run app.py
```
Access the dashboard at `http://localhost:8501`.

## 🧠 Approach

The system combines TF-IDF vectorization with structured features (sleep, stress, energy) to create a high-fidelity input space. It uses a hybrid decision engine where ML models classify the state and a rule-based system determines the best action/timing.

### Label Noise Handling
We implement post-prediction confidence-based flagging. Samples below a threshold (0.6) are logged for review and user-facing messages are automatically hedged (e.g., *"I might be mistaken, but it seems like..."*).

---

Developed for the ArvyaX assignment. Under the username **ojassahu29**.
