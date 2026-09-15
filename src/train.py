import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
for p in [current_dir, project_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from data_pipeline import load_and_clean_data
except ImportError:
    from src.data_pipeline import load_and_clean_data

def get_preprocessor(numeric_features, categorical_features):
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    return preprocessor

def train_and_evaluate():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    csv_path = os.path.join(project_root, 'credit_risk_dataset.csv')
    models_dir = os.path.join(project_root, 'models')
    os.makedirs(models_dir, exist_ok=True)

    print(">>> 1. Loading & Cleaning Data...")
    df = load_and_clean_data(csv_path)
    print(f"Cleaned dataset shape: {df.shape}")

    # Define target and feature columns
    target_col = 'loan_status'
    X = df.drop(columns=[target_col])
    y = df[target_col]

    categorical_features = [
        'person_home_ownership',
        'loan_intent',
        'loan_grade',
        'cb_person_default_on_file'
    ]
    numeric_features = [col for col in X.columns if col not in categorical_features]

    print(f"Numeric features ({len(numeric_features)}): {numeric_features}")
    print(f"Categorical features ({len(categorical_features)}): {categorical_features}")

    # Stratified Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

    preprocessor = get_preprocessor(numeric_features, categorical_features)

    # Candidate models
    candidate_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42, class_weight='balanced', n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, max_depth=6, random_state=42)
    }

    results = {}
    fitted_pipelines = {}

    print("\n>>> 2. Training and Evaluating Candidate Models...")
    for model_name, clf in candidate_models.items():
        print(f"--> Training {model_name}...")
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])

        pipeline.fit(X_train, y_train)
        fitted_pipelines[model_name] = pipeline

        # Predictions
        y_pred = pipeline.predict(X_test)
        if hasattr(pipeline, "predict_proba"):
            y_proba = pipeline.predict_proba(X_test)[:, 1]
        else:
            y_proba = pipeline.decision_function(X_test)

        # Metrics
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        roc_auc = float(roc_auc_score(y_test, y_proba))

        results[model_name] = {
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1_Score": round(f1, 4),
            "ROC_AUC": round(roc_auc, 4)
        }
        print(f"    Results for {model_name}: Accuracy={acc:.4f}, F1={f1:.4f}, ROC-AUC={roc_auc:.4f}")

    # Select best model by ROC_AUC
    best_model_name = max(results, key=lambda m: results[m]["ROC_AUC"])
    best_pipeline = fitted_pipelines[best_model_name]
    print(f"\n>>> 3. Best Model Selected: {best_model_name} (ROC-AUC: {results[best_model_name]['ROC_AUC']})")

    # Save benchmark comparison
    comparison_path = os.path.join(models_dir, 'model_comparison.json')
    with open(comparison_path, 'w') as f:
        json.dump({
            "best_model": best_model_name,
            "metrics": results
        }, f, indent=4)
    print(f"Saved model comparison to: {comparison_path}")

    # Save artifact bundle with pipeline and metadata
    artifact_bundle = {
        "model_name": best_model_name,
        "pipeline": best_pipeline,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "metrics": results[best_model_name],
        "all_metrics": results
    }
    model_save_path = os.path.join(models_dir, 'best_model.joblib')
    joblib.dump(artifact_bundle, model_save_path)
    print(f"Saved best model pipeline to: {model_save_path}")

    return best_model_name, results

if __name__ == '__main__':
    train_and_evaluate()
