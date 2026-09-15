# Real-Time Credit Risk Monitoring with Machine Learning

A full-stack end-to-end Machine Learning web application designed to evaluate loan applications and predict borrower default risk in real-time. Supports both **Flask** and **Streamlit** interfaces.

---

## 📌 Features

- **⚡ Real-Time Risk Assessment**: Evaluates loan applicants instantly and predicts default probability.
- **🎯 Dynamic Tier Classification**: Categorizes risk into **Low**, **Medium**, or **High** with automated business recommendations.
- **📊 Benchmark Comparison**: Evaluates and compares multiple ML models (HistGradientBoosting, Gradient Boosting, Random Forest, Logistic Regression) using ROC-AUC, F1-Score, and Accuracy.
- **🔄 End-to-End Pipeline**: Includes preprocessing, anomaly handling, automated feature engineering, and stratified cross-validation.
- **💻 Dual Interface**:
  - **Flask Web App**: Modern, responsive, glassmorphic UI styled with custom CSS, Chart.js benchmarks, and REST API endpoints (`/api/predict`, `/api/metrics`).
  - **Streamlit Dashboard**: Rapid interactive exploration dashboard.

---

## 📂 Project Structure

```text
├── credit_risk_dataset.csv     # Raw dataset
├── app.py                      # Streamlit web application
├── flask_app.py                # Flask web application & REST API
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore file
├── src/
│   ├── data_pipeline.py        # Data loading, cleaning & feature engineering
│   ├── train.py                # Model training, comparison & serialization
│   └── predictor.py            # Real-time inference engine
├── models/
│   ├── best_model.joblib       # Serialized top-performing ML pipeline
│   └── model_comparison.json   # Model evaluation metrics & benchmark
├── static/
│   ├── css/style.css           # Glassmorphic UI stylesheet
│   └── js/app.js               # Frontend async logic & Chart.js rendering
└── templates/
    └── index.html              # Modern dashboard template
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd "REAL TIME CREDIT RISK MONITORING WITH ML"
```

### 2. Create and activate a virtual environment
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🏃‍♂️ Running the Application

### Option A: Run the Flask Web App (Recommended)
```bash
python flask_app.py
```
Open your browser and navigate to `http://127.0.0.1:5000`.

### Option B: Run the Streamlit App
```bash
streamlit run app.py
```

---

## 📊 API Endpoints (Flask)

- `GET /` - Main web dashboard interface.
- `POST /api/predict` - Real-time inference endpoint (Accepts applicant JSON payload, returns probability, risk tier, action, and key ratios).
- `GET /api/metrics` - Model benchmark performance metrics JSON.
