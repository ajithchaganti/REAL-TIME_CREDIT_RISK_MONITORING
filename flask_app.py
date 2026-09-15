import os
import sys
import json
from flask import Flask, render_template, request, jsonify

# Ensure project root is in Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.predictor import CreditRiskPredictor

app = Flask(__name__)

# ── Load predictor once at startup ──────────────────────────────────────────
predictor = None
load_error = None

try:
    predictor = CreditRiskPredictor()
except Exception as e:
    load_error = str(e)


def _load_benchmark_metrics():
    comp_path = os.path.join(current_dir, 'models', 'model_comparison.json')
    if os.path.exists(comp_path):
        with open(comp_path, 'r') as f:
            return json.load(f)
    return None


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    """Accept loan application JSON, return risk prediction."""
    if predictor is None:
        return jsonify({'error': f'Model not loaded: {load_error}'}), 503

    data = request.get_json(force=True)

    # Validate & cast required fields
    try:
        payload = {
            'person_age':                   int(data['person_age']),
            'person_income':                float(data['person_income']),
            'person_home_ownership':        str(data['person_home_ownership']),
            'person_emp_length':            float(data['person_emp_length']),
            'loan_intent':                  str(data['loan_intent']),
            'loan_grade':                   str(data['loan_grade']),
            'loan_amnt':                    float(data['loan_amnt']),
            'loan_int_rate':                float(data['loan_int_rate']),
            'cb_person_default_on_file':    str(data['cb_person_default_on_file']),
            'cb_person_cred_hist_length':   int(data['cb_person_cred_hist_length']),
        }
    except (KeyError, ValueError, TypeError) as e:
        return jsonify({'error': f'Invalid input: {e}'}), 400

    result = predictor.predict(payload)

    # Compute derived ratios for the frontend
    loan_amnt = payload['loan_amnt']
    income    = payload['person_income']
    int_rate  = payload['loan_int_rate']
    loan_grade = payload['loan_grade']

    dti             = round((loan_amnt / income) * 100, 2) if income > 0 else 0
    annual_interest = round(loan_amnt * (int_rate / 100), 2)

    # Risk drivers
    factors = []
    if dti > 35:
        factors.append("High debt burden relative to applicant income.")
    if payload['cb_person_default_on_file'] == 'Y':
        factors.append("Prior historical default recorded in credit bureau file.")
    if loan_grade in ['D', 'E', 'F', 'G']:
        factors.append(f"Subprime loan grade ({loan_grade}) associated with higher historical default rate.")
    if payload['person_emp_length'] < 2.0:
        factors.append("Limited employment tenure (< 2 years).")
    if not factors:
        factors.append("Strong financial profile: moderate borrowing ratio, solid credit history, clean bureau record.")

    result.update({
        'dti':             dti,
        'annual_interest': annual_interest,
        'loan_grade':      loan_grade,
        'factors':         factors,
    })

    return jsonify(result)


@app.route('/api/metrics', methods=['GET'])
def metrics():
    """Return model benchmark metrics JSON."""
    data = _load_benchmark_metrics()
    if data is None:
        return jsonify({'error': 'Metrics file not found. Run training first.'}), 404
    return jsonify(data)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(debug=debug, host='0.0.0.0', port=port)
