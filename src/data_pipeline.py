import pandas as pd
import numpy as np

def load_and_clean_data(csv_path: str) -> pd.DataFrame:
    """
    Loads credit risk dataset and cleans missing values and unrealistic outliers.
    """
    df = pd.read_csv(csv_path)

    # 1. Outlier removal for unrealistic values
    # In the dataset, age can be > 100 (e.g. 123, 144) and emp_length > 100
    df = df[df['person_age'] <= 90]
    df = df[df['person_emp_length'].isna() | (df['person_emp_length'] <= 50)]
    
    # Also person_emp_length cannot logically exceed person_age - 14
    df = df[df['person_emp_length'].isna() | (df['person_emp_length'] <= (df['person_age'] - 14))]

    # 2. Impute missing values
    # Median employment length by age bracket or overall
    median_emp = df['person_emp_length'].median()
    df['person_emp_length'] = df['person_emp_length'].fillna(median_emp)

    # Impute missing interest rates grouped by loan_grade
    df['loan_int_rate'] = df.groupby('loan_grade')['loan_int_rate'].transform(
        lambda x: x.fillna(x.median())
    )
    # If any remain, fill with overall median
    if df['loan_int_rate'].isna().sum() > 0:
        df['loan_int_rate'] = df['loan_int_rate'].fillna(df['loan_int_rate'].median())

    # 3. Feature Engineering
    df = add_engineered_features(df)

    return df

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies domain-specific financial risk feature engineering.
    """
    df = df.copy()

    # Loan to income ratio (safety clip to avoid div by zero)
    income_safe = df['person_income'].replace(0, np.nan).fillna(1.0)
    
    # Recalculate or ensure loan_percent_income is consistent
    df['loan_percent_income'] = (df['loan_amnt'] / income_safe).clip(0, 1.5)

    # Estimated annual interest burden relative to income
    annual_interest_cost = df['loan_amnt'] * (df['loan_int_rate'] / 100.0)
    df['interest_burden'] = (annual_interest_cost / income_safe).clip(0, 1.0)

    # Credit history relative to age (proportion of life with credit)
    age_safe = df['person_age'].replace(0, 18)
    df['cred_hist_to_age_ratio'] = (df['cb_person_cred_hist_length'] / age_safe).clip(0, 1.0)

    # Employment stability factor: loan amount per year of employment
    emp_safe = df['person_emp_length'].replace(0, 0.5)
    df['loan_to_emp_ratio'] = df['loan_amnt'] / emp_safe

    # Flag for high risk intent categories
    high_risk_intents = ['DEBTCONSOLIDATION', 'MEDICAL']
    df['is_high_risk_intent'] = df['loan_intent'].isin(high_risk_intents).astype(int)

    return df
