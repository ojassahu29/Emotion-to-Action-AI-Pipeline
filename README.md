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

## 🛠️ Getting Started (Beginner Friendly Guide)

Follow these steps to get the project running on your local machine.

### Prerequisites
- **Python 3.9+** installed on your system.
- Basic familiarity with the command line / terminal.

### 1. Set Up a Virtual Environment (Recommended)
It's best practice to create a virtual environment so dependencies don't conflict with other projects.
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
Installs all required libraries (like pandas, scikit-learn, FastAPI, and Streamlit).
```bash
pip install -r requirements.txt
```

### 3. Run the Full ML Pipeline (Batch Inference)
This runs the core pipeline: it loads the data, cleans text, extracts features, runs the models, and generates an output CSV with predictions and supportive messages.
```bash
python main.py
```
*Outputs are saved to:*
- `outputs/predictions.csv` (contains the predicted emotion, intensity, action, and message)
- `outputs/low_confidence_log.csv` (contains low-confidence predictions flagged for review)

### 4. Try the Web Dashboard (Streamlit UI)
Want a visual, interactive way to test the models? We built a beautiful web dashboard!
```bash
streamlit run app.py
```
*This will automatically open a new tab in your browser at `http://localhost:8501`. Just type a journal entry, adjust the sliders, and click "Analyze"!*

### 5. Start the Local API (FastAPI)
Want to connect this AI to another app or frontend? Start the local API server.
```bash
uvicorn src.api.app:app --port 8000
```
*You can view the interactive developer documentation at `http://localhost:8000/docs`.*

## 🧠 Approach

The system combines TF-IDF vectorization with structured features (sleep, stress, energy) to create a high-fidelity input space. It uses a hybrid decision engine where ML models classify the state and a rule-based system determines the best action/timing.

### Label Noise Handling
We implement post-prediction confidence-based flagging. Samples below a threshold (0.6) are logged for review and user-facing messages are automatically hedged (e.g., *"I might be mistaken, but it seems like..."*).

---

Developed for the ArvyaX assignment. Under the username **ojassahu29**.
 
 
 
