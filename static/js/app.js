/* ═══════════════════════════════════════════════════════════════════════════
   REAL-TIME CREDIT RISK MONITOR — App Logic
   ═══════════════════════════════════════════════════════════════════════════ */

// ── Footer year ────────────────────────────────────────────────────────────
document.getElementById('footer-year').textContent = new Date().getFullYear();

// ── Tab Navigation ──────────────────────────────────────────────────────────
const tabBtns   = document.querySelectorAll('.tab-btn');
const tabPanels = document.querySelectorAll('.tab-panel');

tabBtns.forEach(btn => {
  btn.addEventListener('click', () => {
    const target = btn.dataset.tab;

    tabBtns.forEach(b => {
      b.classList.toggle('active', b === btn);
      b.setAttribute('aria-selected', b === btn ? 'true' : 'false');
    });

    tabPanels.forEach(p => {
      p.classList.toggle('active', p.id === `panel-${target}`);
    });

    // Lazy-load metrics tab when first opened
    if (target === 'metrics') loadMetrics();
  });
});

// ── Preset Definitions ──────────────────────────────────────────────────────
const PRESETS = {
  prime: {
    person_age: 34, person_income: 115000, person_home_ownership: 'MORTGAGE',
    person_emp_length: 8.0, loan_intent: 'HOMEIMPROVEMENT', loan_grade: 'A',
    loan_amnt: 12000, loan_int_rate: 7.5, cb_person_default_on_file: 'N',
    cb_person_cred_hist_length: 9
  },
  moderate: {
    person_age: 25, person_income: 42000, person_home_ownership: 'RENT',
    person_emp_length: 2.0, loan_intent: 'PERSONAL', loan_grade: 'C',
    loan_amnt: 14000, loan_int_rate: 13.5, cb_person_default_on_file: 'N',
    cb_person_cred_hist_length: 3
  },
  high: {
    person_age: 22, person_income: 18000, person_home_ownership: 'RENT',
    person_emp_length: 1.0, loan_intent: 'DEBTCONSOLIDATION', loan_grade: 'F',
    loan_amnt: 25000, loan_int_rate: 21.0, cb_person_default_on_file: 'Y',
    cb_person_cred_hist_length: 2
  }
};

// Apply preset to form fields
function applyPreset(presetKey) {
  const preset = PRESETS[presetKey];
  Object.entries(preset).forEach(([key, val]) => {
    const el = document.getElementById(key) || document.querySelector(`[name="${key}"]`);
    if (el) el.value = val;
  });

  // Update preset button active states
  document.querySelectorAll('.preset-btn').forEach(btn => {
    const isActive = btn.dataset.preset === presetKey;
    btn.classList.toggle('active', isActive);
    // Re-apply correct class
    btn.classList.remove('preset-prime', 'preset-moderate', 'preset-high');
    btn.classList.add(`preset-${btn.dataset.preset}`);
  });
}

// Attach preset buttons
document.querySelectorAll('.preset-btn').forEach(btn => {
  btn.addEventListener('click', () => applyPreset(btn.dataset.preset));
});

// Apply the default preset on page load
applyPreset('prime');

// ── Form Submission ─────────────────────────────────────────────────────────
const form       = document.getElementById('loan-form');
const submitBtn  = document.getElementById('submit-btn');
const btnText    = submitBtn.querySelector('.btn-text');
const btnSpinner = document.getElementById('btn-spinner');

form.addEventListener('submit', async (e) => {
  e.preventDefault();

  // Build payload from form
  const fd = new FormData(form);
  const payload = {
    person_age:                 Number(fd.get('person_age')),
    person_income:              Number(fd.get('person_income')),
    person_home_ownership:      fd.get('person_home_ownership'),
    person_emp_length:          Number(fd.get('person_emp_length')),
    loan_intent:                fd.get('loan_intent'),
    loan_grade:                 fd.get('loan_grade'),
    loan_amnt:                  Number(fd.get('loan_amnt')),
    loan_int_rate:              Number(fd.get('loan_int_rate')),
    cb_person_default_on_file:  fd.get('cb_person_default_on_file'),
    cb_person_cred_hist_length: Number(fd.get('cb_person_cred_hist_length')),
  };

  setSubmitting(true);
  hideError();

  try {
    const res  = await fetch('/api/predict', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(payload)
    });
    const data = await res.json();

    if (!res.ok) throw new Error(data.error || `Server error ${res.status}`);
    renderResult(data);
  } catch (err) {
    showError(err.message);
  } finally {
    setSubmitting(false);
  }
});

function setSubmitting(loading) {
  submitBtn.disabled = loading;
  btnText.classList.toggle('hidden', loading);
  btnSpinner.classList.toggle('hidden', !loading);
}

// ── Render Result Card ──────────────────────────────────────────────────────
function renderResult(data) {
  const placeholder = document.getElementById('result-placeholder');
  const card        = document.getElementById('result-card');
  const errorCard   = document.getElementById('error-card');

  placeholder.classList.add('hidden');
  errorCard.classList.add('hidden');
  card.classList.remove('hidden');

  // Risk color map
  const colorMap = {
    LOW:    '#10b981',
    MEDIUM: '#f59e0b',
    HIGH:   '#ef4444'
  };
  const riskColor = colorMap[data.risk_level] || data.risk_color || '#6366f1';

  // Card border accent
  card.style.borderLeftColor = riskColor;

  // Badge
  const badge = document.getElementById('risk-badge');
  badge.textContent = `${data.risk_level} RISK`;
  badge.className   = `badge-pill badge-${data.risk_level}`;

  // Score circle
  const scoreCircle = document.getElementById('score-circle');
  scoreCircle.textContent  = `${data.default_probability_percent}%`;
  scoreCircle.style.color  = riskColor;

  // Health score
  document.getElementById('credit-health-label').textContent =
    `Credit Health Score: ${data.credit_health_score} / 100`;

  // Progress bar  (fill = default probability)
  const barFill = document.getElementById('score-bar-fill');
  barFill.style.background = riskColor;
  // Trigger animation after brief delay
  setTimeout(() => {
    barFill.style.width = `${data.default_probability_percent}%`;
  }, 80);

  // Action & recommendation
  const actionEl = document.getElementById('action-label');
  actionEl.textContent  = `ACTION: ${data.action}`;
  actionEl.style.color  = riskColor;

  document.getElementById('recommendation-text').textContent = data.recommendation;

  // DTI
  const dti = data.dti;
  document.getElementById('dti-value').textContent = `${dti.toFixed(1)}%`;
  document.getElementById('dti-delta').textContent =
    dti <= 20 ? '✅ Optimal (< 20%)' : dti <= 40 ? '⚠️ Elevated' : '🔴 High Risk (> 40%)';

  // Annual interest
  document.getElementById('interest-value').textContent =
    `$${data.annual_interest.toLocaleString('en-US', { maximumFractionDigits: 0 })}/yr`;
  document.getElementById('interest-delta').textContent = `Grade ${data.loan_grade} Rate`;

  // Factors
  const ul = document.getElementById('factors-list');
  ul.innerHTML = '';
  const isPositive = data.factors.length === 1 && data.factors[0].startsWith('Strong');
  data.factors.forEach(f => {
    const li = document.createElement('li');
    li.textContent = (isPositive ? '✅ ' : '⚠️ ') + f;
    ul.appendChild(li);
  });

  // Model caption
  document.getElementById('model-caption').textContent =
    `🤖 Powered by ${data.model_used} Pipeline`;
}

// ── Error helpers ───────────────────────────────────────────────────────────
function showError(msg) {
  const placeholder = document.getElementById('result-placeholder');
  const card        = document.getElementById('result-card');
  const errorCard   = document.getElementById('error-card');

  placeholder.classList.add('hidden');
  card.classList.add('hidden');
  errorCard.classList.remove('hidden');
  document.getElementById('error-text').textContent = msg;
}

function hideError() {
  document.getElementById('error-card').classList.add('hidden');
}

// ── Metrics Tab ─────────────────────────────────────────────────────────────
let metricsLoaded  = false;
let metricsChart   = null;

async function loadMetrics() {
  if (metricsLoaded) return;

  const loading = document.getElementById('metrics-loading');
  const content = document.getElementById('metrics-content');
  const errEl   = document.getElementById('metrics-error');
  const errText = document.getElementById('metrics-error-text');

  loading.classList.remove('hidden');
  content.classList.add('hidden');
  errEl.classList.add('hidden');

  try {
    const res  = await fetch('/api/metrics');
    const data = await res.json();

    if (!res.ok) throw new Error(data.error || `Server error ${res.status}`);

    renderMetrics(data);
    loading.classList.add('hidden');
    content.classList.remove('hidden');
    metricsLoaded = true;
  } catch (err) {
    loading.classList.add('hidden');
    errEl.classList.remove('hidden');
    errText.textContent = err.message;
  }
}

function renderMetrics(data) {
  const metricsDict = data.metrics || {};
  const bestName    = data.best_model || '';

  const COLS = ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC'];

  // Find per-column maxima
  const maxVals = {};
  COLS.forEach(col => {
    maxVals[col] = Math.max(...Object.values(metricsDict).map(m => m[col] || 0));
  });

  // Build table rows
  const tbody = document.getElementById('metrics-tbody');
  tbody.innerHTML = '';

  Object.entries(metricsDict).forEach(([modelName, m]) => {
    const tr = document.createElement('tr');
    if (modelName === bestName) tr.classList.add('best-row');

    const tdName = document.createElement('td');
    tdName.classList.add('model-name-cell');
    tdName.textContent = modelName;
    tr.appendChild(tdName);

    COLS.forEach(col => {
      const td = document.createElement('td');
      const val = (m[col] ?? 0).toFixed(4);
      td.textContent = val;
      if (m[col] === maxVals[col]) td.classList.add('best-cell');
      tr.appendChild(td);
    });

    tbody.appendChild(tr);
  });

  // Best model banner
  const banner = document.getElementById('best-model-banner');
  if (bestName && metricsDict[bestName]) {
    const auc = metricsDict[bestName].ROC_AUC?.toFixed(4) ?? '—';
    banner.innerHTML = `🌟 <strong>Best Selected Model: ${bestName}</strong> — achieved highest discriminative ability with ROC-AUC: ${auc}`;
  }

  // Chart
  const labels  = Object.keys(metricsDict);
  const metrics = ['ROC_AUC', 'F1_Score', 'Accuracy'];

  const CHART_COLORS = [
    { border: '#6366f1', bg: 'rgba(99,102,241,0.7)' },
    { border: '#10b981', bg: 'rgba(16,185,129,0.7)' },
    { border: '#f59e0b', bg: 'rgba(245,158,11,0.7)' },
  ];

  const datasets = metrics.map((metric, i) => ({
    label:           metric.replace('_', '-'),
    data:            labels.map(l => metricsDict[l]?.[metric] ?? 0),
    backgroundColor: CHART_COLORS[i].bg,
    borderColor:     CHART_COLORS[i].border,
    borderWidth:     1.5,
    borderRadius:    4,
  }));

  const ctx = document.getElementById('metrics-chart').getContext('2d');
  if (metricsChart) metricsChart.destroy();

  metricsChart = new Chart(ctx, {
    type: 'bar',
    data: { labels, datasets },
    options: {
      responsive: true,
      animation: { duration: 900, easing: 'easeOutQuart' },
      plugins: {
        legend: {
          labels: { color: '#94a3b8', font: { family: 'Plus Jakarta Sans', size: 12 } }
        },
        title: {
          display: true,
          text:    'Model Evaluation Metric Comparison (Test Set)',
          color:   '#e2e8f0',
          font:    { family: 'Plus Jakarta Sans', size: 14, weight: '700' },
          padding: { bottom: 20 }
        },
        tooltip: {
          backgroundColor: '#111d35',
          borderColor:     'rgba(255,255,255,0.08)',
          borderWidth:     1,
          titleColor:      '#f0f4ff',
          bodyColor:       '#94a3b8',
          titleFont:       { family: 'Plus Jakarta Sans', weight: '700' },
          bodyFont:        { family: 'Plus Jakarta Sans' },
          callbacks: {
            label: ctx => ` ${ctx.dataset.label}: ${ctx.parsed.y.toFixed(4)}`
          }
        }
      },
      scales: {
        x: {
          grid:  { color: 'rgba(255,255,255,0.04)' },
          ticks: { color: '#94a3b8', font: { family: 'Plus Jakarta Sans', size: 12 } }
        },
        y: {
          min:   0.6,
          max:   1.0,
          grid:  { color: 'rgba(255,255,255,0.04)' },
          ticks: {
            color:    '#94a3b8',
            font:     { family: 'Plus Jakarta Sans', size: 12 },
            callback: v => v.toFixed(2)
          }
        }
      }
    }
  });
}
