import os
import sys
import pandas as pd

# Add src to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_pipeline import load_and_clean_data
from predictor import CreditRiskPredictor

def main():
    print("=" * 60)
    print("REAL-TIME CREDIT RISK MONITORING SYSTEM - BATCH 05")
    print("=" * 60)

    # 1. Load and display raw vs cleaned
    csv_path = os.path.join(os.path.dirname(__file__), "credit_risk_dataset.csv")
    print("\n[1] Loading dataset...")
    df_clean = load_and_clean_data(csv_path)
    print(f"Cleaned dataset shape: {df_clean.shape}")
    print("\nCleaned Sample Data:")
    print(df_clean[['person_age', 'person_income', 'loan_amnt', 'loan_grade', 'loan_status', 'loan_percent_income', 'interest_burden']].head())

    # 2. Risk Evaluation Demonstration
    print("\n[2] Running Inference Test via Predictor...")
    predictor = CreditRiskPredictor()
    
    sample_candidate = {
        "person_age": 28,
        "person_income": 55000,
        "person_home_ownership": "RENT",
        "person_emp_length": 3.0,
        "loan_intent": "EDUCATION",
        "loan_grade": "B",
        "loan_amnt": 8000,
        "loan_int_rate": 10.5,
        "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 4
    }
    
    result = predictor.predict(sample_candidate)
    print("\nEvaluation Output:")
    print(f"  • Model Selected        : {result['model_used']}")
    print(f"  • Default Probability   : {result['default_probability_percent']}%")
    print(f"  • Risk Level            : {result['risk_level']} RISK")
    print(f"  • Credit Health Score   : {result['credit_health_score']} / 100")
    print(f"  • Action Recommendation : {result['action']} - {result['recommendation']}")

    print("\n[3] Web Application:")
    print("  To launch the interactive dashboard, run:")
    print("  python -m streamlit run app.py")
    print("=" * 60)

if __name__ == "__main__":
    main()
