import os
import sys
import json
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# Ensure project root is in Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.predictor import CreditRiskPredictor
 
# Page Configuration
st.set_page_config(
    page_title="Real-Time Credit Risk Monitor",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern glassmorphic and executive design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Gradient header container */
    .hero-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 26px 32px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }
    
    .hero-title {
        color: #f8fafc;
        font-size: 28px;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .hero-subtitle {
        color: #94a3b8;
        font-size: 14px;
        margin-top: 6px;
        margin-bottom: 0;
    }

    /* Result Card styling */
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 18px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 24px -6px rgba(0, 0, 0, 0.4);
    }

    .badge-pill {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 14px;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }

    .badge-low {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }

    .badge-med {
        background-color: rgba(245, 158, 11, 0.15);
        color: #F59E0B;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }

    .badge-high {
        background-color: rgba(239, 68, 68, 0.15);
        color: #EF4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }

    .score-circle {
        font-size: 42px;
        font-weight: 800;
        line-height: 1;
        margin: 10px 0;
    }

    /* Form container */
    .form-panel {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Header Section
st.markdown("""
<div class="hero-container">
    <h1 class="hero-title">💳 Real-Time Credit Risk Monitoring System</h1>
    <p class="hero-subtitle">Machine Learning Powered Credit Scoring • Instant Default Probability • Automated Tier Assessment</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def get_predictor():
    return CreditRiskPredictor()

@st.cache_data
def get_benchmark_metrics():
    comp_path = os.path.join(current_dir, 'models', 'model_comparison.json')
    if os.path.exists(comp_path):
        with open(comp_path, 'r') as f:
            return json.load(f)
    return None

try:
    predictor = get_predictor()
    benchmark_data = get_benchmark_metrics()
except Exception as e:
    st.error(f"Error loading prediction models: {e}. Please ensure model training is complete.")
    st.stop()

# Tabs
tab_assess, tab_metrics, tab_pipeline = st.tabs([
    "⚡ Instant Risk Assessment",
    "📊 Model Comparison & Metrics",
    "🔄 End-to-End Pipeline Architecture"
])

# ================= TAB 1: Instant Risk Assessment =================
with tab_assess:
    col_input, col_result = st.columns([1.1, 1.0], gap="large")

    with col_input:
        st.markdown("### 👤 Applicant & Loan Details")

        # Quick preset buttons for demo convenience
        st.markdown("<p style='color: #94a3b8; font-size: 13px;'>Quick Demo Presets:</p>", unsafe_allow_html=True)
        p_col1, p_col2, p_col3 = st.columns(3)
        
        # Session state for form presets
        if 'preset' not in st.session_state:
            st.session_state.preset = "prime"

        if p_col1.button("🟢 Prime Borrower"):
            st.session_state.preset = "prime"
        if p_col2.button("🟡 Moderate Risk"):
            st.session_state.preset = "moderate"
        if p_col3.button("🔴 High Risk Default"):
            st.session_state.preset = "high"

        # Preset values
        if st.session_state.preset == "prime":
            def_age, def_inc, def_home, def_emp = 34, 115000, "MORTGAGE", 8.0
            def_intent, def_grade, def_amnt, def_rate = "HOMEIMPROVEMENT", "A", 12000, 7.5
            def_default, def_cred_hist = "N", 9
        elif st.session_state.preset == "moderate":
            def_age, def_inc, def_home, def_emp = 25, 42000, "RENT", 2.0
            def_intent, def_grade, def_amnt, def_rate = "PERSONAL", "C", 14000, 13.5
            def_default, def_cred_hist = "N", 3
        else: # high
            def_age, def_inc, def_home, def_emp = 22, 18000, "RENT", 1.0
            def_intent, def_grade, def_amnt, def_rate = "DEBTCONSOLIDATION", "F", 25000, 21.0
            def_default, def_cred_hist = "Y", 2

        with st.form("loan_application_form"):
            st.markdown("##### 1. Demographics & Financials")
            c1, c2 = st.columns(2)
            with c1:
                age = st.number_input("Age (Years)", min_value=18, max_value=85, value=def_age, step=1)
                home_ownership = st.selectbox(
                    "Home Ownership",
                    options=["RENT", "MORTGAGE", "OWN", "OTHER"],
                    index=["RENT", "MORTGAGE", "OWN", "OTHER"].index(def_home)
                )
            with c2:
                income = st.number_input("Annual Income ($)", min_value=1000, max_value=2000000, value=def_inc, step=5000)
                emp_length = st.number_input("Employment Length (Years)", min_value=0.0, max_value=45.0, value=float(def_emp), step=0.5)

            st.markdown("##### 2. Loan Specifics")
            c3, c4 = st.columns(2)
            with c3:
                loan_amnt = st.number_input("Requested Loan Amount ($)", min_value=500, max_value=100000, value=def_amnt, step=1000)
                loan_intent = st.selectbox(
                    "Loan Intent",
                    options=["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"],
                    index=["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"].index(def_intent)
                )
            with c4:
                loan_grade = st.selectbox(
                    "Loan Grade",
                    options=["A", "B", "C", "D", "E", "F", "G"],
                    index=["A", "B", "C", "D", "E", "F", "G"].index(def_grade)
                )
                loan_int_rate = st.number_input("Interest Rate (%)", min_value=4.0, max_value=35.0, value=float(def_rate), step=0.25)

            st.markdown("##### 3. Credit History")
            c5, c6 = st.columns(2)
            with c5:
                default_on_file = st.selectbox(
                    "Historical Default on File?",
                    options=["N", "Y"],
                    index=["N", "Y"].index(def_default)
                )
            with c6:
                cred_hist_length = st.number_input("Credit History Length (Years)", min_value=0, max_value=40, value=def_cred_hist, step=1)

            submit_btn = st.form_submit_button("⚡ Run Real-Time Credit Risk Evaluation", use_container_width=True)

    with col_result:
        st.markdown("### 🎯 Real-Time Decision & Risk Profile")

        input_payload = {
            "person_age": age,
            "person_income": income,
            "person_home_ownership": home_ownership,
            "person_emp_length": emp_length,
            "loan_intent": loan_intent,
            "loan_grade": loan_grade,
            "loan_amnt": loan_amnt,
            "loan_int_rate": loan_int_rate,
            "cb_person_default_on_file": default_on_file,
            "cb_person_cred_hist_length": cred_hist_length
        }

        result = predictor.predict(input_payload)

        # Style badges
        risk_level = result["risk_level"]
        badge_class = "badge-low" if risk_level == "LOW" else ("badge-med" if risk_level == "MEDIUM" else "badge-high")
        
        # Risk card
        st.markdown(f"""
        <div class="metric-card" style="border-left: 5px solid {result['risk_color']};">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 14px; font-weight: 600; color: #94a3b8;">OVERALL RISK TIER</span>
                <span class="badge-pill {badge_class}">{risk_level} RISK</span>
            </div>
            <div style="display: flex; align-items: baseline; gap: 15px; margin-top: 14px;">
                <div class="score-circle" style="color: {result['risk_color']};">
                    {result['default_probability_percent']}%
                </div>
                <div style="color: #94a3b8; font-size: 14px;">
                    Estimated Default Probability<br>
                    <strong style="color: #f1f5f9;">Credit Health Score: {result['credit_health_score']} / 100</strong>
                </div>
            </div>
            <hr style="border: 0; border-top: 1px solid rgba(255, 255, 255, 0.1); margin: 16px 0;">
            <div style="font-size: 15px; font-weight: 700; color: {result['risk_color']}; margin-bottom: 4px;">
                ACTION: {result['action']}
            </div>
            <div style="font-size: 13.5px; color: #cbd5e1; line-height: 1.5;">
                {result['recommendation']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Key Financial Ratios Breakdown
        st.markdown("#### 📐 Evaluated Financial Ratios")
        dti = (loan_amnt / income) * 100 if income > 0 else 0
        annual_interest = loan_amnt * (loan_int_rate / 100)

        r_col1, r_col2 = st.columns(2)
        with r_col1:
            st.metric(
                label="Debt-to-Income Ratio (DTI)",
                value=f"{dti:.1f}%",
                delta="Optimal (< 20%)" if dti <= 20 else ("Elevated" if dti <= 40 else "High Risk (> 40%)"),
                delta_color="normal" if dti <= 20 else "inverse"
            )
        with r_col2:
            st.metric(
                label="Annual Interest Obligation",
                value=f"${annual_interest:,.0f}/yr",
                delta=f"Grade {loan_grade} Rate"
            )

        # Credit factors summary
        st.markdown("#### 🔍 Primary Risk Drivers")
        factors = []
        if dti > 35:
            factors.append("⚠️ High debt burden relative to applicant income.")
        if default_on_file == "Y":
            factors.append("⚠️ Prior historical default recorded in credit bureau file.")
        if loan_grade in ["D", "E", "F", "G"]:
            factors.append(f"⚠️ Subprime loan grade ({loan_grade}) associated with higher historical default rate.")
        if emp_length < 2.0:
            factors.append("ℹ️ Limited employment tenure (< 2 years).")
        if not factors:
            factors.append("✅ Strong financial profile: moderate borrowing ratio, solid credit history, clean bureau record.")

        for f in factors:
            st.markdown(f"- {f}")

        st.caption(f"🤖 Powered by {result['model_used']} Pipeline")


# ================= TAB 2: Model Comparison & Benchmarks =================
with tab_metrics:
    st.markdown("### 🏆 Machine Learning Model Benchmark Comparison")
    st.write("Each candidate model was trained and evaluated on an independent 20% stratified test split:")

    if benchmark_data and "metrics" in benchmark_data:
        metrics_dict = benchmark_data["metrics"]
        best_name = benchmark_data.get("best_model", "")

        df_metrics = pd.DataFrame.from_dict(metrics_dict, orient="index")
        df_metrics.index.name = "Model"
        df_metrics = df_metrics.reset_index()

        # Display formatted comparison dataframe
        st.dataframe(
            df_metrics.style.highlight_max(
                subset=["Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC"],
                color="#064e3b"
            ).format({
                "Accuracy": "{:.4f}",
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1_Score": "{:.4f}",
                "ROC_AUC": "{:.4f}"
            }),
            use_container_width=True
        )

        st.success(f"🌟 **Best Selected Model:** **{best_name}** achieved the highest discriminative ability with **ROC-AUC: {metrics_dict[best_name]['ROC_AUC']}**")

        # Interactive Chart for ROC-AUC and F1 Score
        chart_data = pd.melt(
            df_metrics,
            id_vars=["Model"],
            value_vars=["ROC_AUC", "F1_Score", "Accuracy"],
            var_name="Metric",
            value_name="Score"
        )

        chart = alt.Chart(chart_data).mark_bar().encode(
            x=alt.X('Model:N', title="Model Architecture"),
            y=alt.Y('Score:Q', scale=alt.Scale(domain=[0.6, 1.0]), title="Score"),
            color=alt.Color('Metric:N', scale=alt.Scale(scheme='tableau10')),
            xOffset='Metric:N',
            tooltip=['Model', 'Metric', 'Score']
        ).properties(
            title="Model Evaluation Metric Comparison (Test Set)",
            height=380
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Training metrics are being compiled...")


# ================= TAB 3: Pipeline Architecture =================
with tab_pipeline:
    st.markdown("### 🔄 End-to-End ML Pipeline Architecture")
    st.markdown("""
    The system follows a strict modular design guaranteeing reproducibility, data hygiene, and sub-second inference:

    | Stage | Operations Performed |
    | :--- | :--- |
    | **1. Kaggle Dataset** | Loaded raw dataset containing 32,581 loan records across 12 behavioral/financial attributes. |
    | **2. Data Cleaning** | Filtered impossible anomalies (ages > 90, employment tenure > 50 yrs). Imputed missing values with median values grouped by loan grade. |
    | **3. Exploratory Analysis** | Assessed distribution of target `loan_status` (imbalanced default distribution) and multi-collinearity. |
    | **4. Feature Engineering** | Constructed Domain Ratios: `loan_percent_income`, `interest_burden`, `cred_hist_to_age_ratio`, `loan_to_emp_ratio`, and `is_high_risk_intent`. |
    | **5. Data Preprocessing** | Built Scikit-Learn `ColumnTransformer`: `StandardScaler` for numeric columns, `OneHotEncoder` for categoricals. |
    | **6. Train/Test Split** | 80/20 Stratified Split preserving the true proportion of loan defaults across splits. |
    | **7. ML Models** | Trained Logistic Regression, Random Forest, Gradient Boosting, and HistGradientBoosting classifiers. |
    | **8. Benchmark & Selection** | Ranked candidates by ROC-AUC and F1-score; selected and serialized the top-performing pipeline. |
    | **9. Risk Probability & Tiers** | Mapped raw probabilities into actionable business risk tiers: Low (<25%), Medium (25-60%), High (≥60%). |
    | **10. Streamlit Web App** | Delivered a real-time reactive monitoring dashboard for live loan underwriting and credit analysis. |
    """)
