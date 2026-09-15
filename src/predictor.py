import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
for p in [current_dir, project_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from data_pipeline import add_engineered_features
except ImportError:
    from src.data_pipeline import add_engineered_features

class CreditRiskPredictor:
    def __init__(self, model_path: str = None):
        if model_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            model_path = os.path.join(project_root, 'models', 'best_model.joblib')
        
        self.model_path = model_path
        self.bundle = None
        self.pipeline = None
        self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file not found at {self.model_path}. Please run src/train.py first."
            )
        self.bundle = joblib.load(self.model_path)
        self.pipeline = self.bundle['pipeline']
        self.model_name = self.bundle.get('model_name', 'Unknown')
        self.metrics = self.bundle.get('metrics', {})

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Takes raw customer/loan inputs, computes engineered features,
        and outputs probability, risk tier, and lending recommendations.
        """
        # Convert dictionary to DataFrame
        df = pd.DataFrame([input_data])

        # Apply feature engineering
        df_processed = add_engineered_features(df)

        # Get probability of default (Class 1)
        proba_default = float(self.pipeline.predict_proba(df_processed)[0, 1])
        prediction = int(proba_default >= 0.5)

        # Risk Tier Classification
        if proba_default < 0.25:
            risk_level = "LOW"
            color = "#10B981" # Green
            recommendation = "Approved: Standard or prime interest rates applicable."
            action = "APPROVE"
        elif proba_default < 0.60:
            risk_level = "MEDIUM"
            color = "#F59E0B" # Amber
            recommendation = "Conditional: Requires manual credit assessment, verification of income, or co-signer."
            action = "MANUAL REVIEW"
        else:
            risk_level = "HIGH"
            color = "#EF4444" # Red
            recommendation = "High Risk: High probability of default. Decline application or require substantial collateral."
            action = "DECLINE"

        # Calculate credit health score (0 - 100, where 100 is best / lowest risk)
        credit_score = int(round((1.0 - proba_default) * 100))

        return {
            "default_probability": round(proba_default, 4),
            "default_probability_percent": round(proba_default * 100, 2),
            "prediction": prediction,
            "risk_level": risk_level,
            "risk_color": color,
            "action": action,
            "recommendation": recommendation,
            "credit_health_score": credit_score,
            "model_used": self.model_name
        }

if __name__ == "__main__":
    predictor = CreditRiskPredictor()
    sample_input = {
        "person_age": 25,
        "person_income": 60000,
        "person_home_ownership": "MORTGAGE",
        "person_emp_length": 4.0,
        "loan_intent": "EDUCATION",
        "loan_grade": "B",
        "loan_amnt": 10000,
        "loan_int_rate": 11.2,
        "loan_percent_income": 0.16,
        "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 3
    }
    res = predictor.predict(sample_input)
    print("Sample Prediction Result:", res)
