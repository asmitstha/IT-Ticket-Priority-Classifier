"""
IT Ticket Priority Classifier — Flask Web App
Loads the saved Random Forest model and serves predictions via a clean web UI.
"""

from flask import Flask, render_template, request, jsonify
import joblib
import re
import numpy as np
from scipy.sparse import hstack, csr_matrix
import os

app = Flask(__name__)

# ── Load saved model ──────────────────────────────────────────────────────────
# Place your saved model file (ticket_priority_FINAL_88pct.joblib) in the same
# folder as this app.py file, then run: python app.py
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'ticket_priority_FINAL.joblib')

model_loaded = False
model = tfidf = le_type = le_queue = None

try:
    saved      = joblib.load(MODEL_PATH)
    model      = saved['model']
    tfidf      = saved['tfidf']
    le_type    = saved['le_type']
    le_queue   = saved['le_queue']
    model_loaded = True
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"⚠️  Model not found at {MODEL_PATH}")
    print(f"   Error: {e}")
    print("   Running in DEMO mode — predictions will be simulated.")

# ── Urgency keywords ──────────────────────────────────────────────────────────
URGENCY_WORDS = {
    'urgent': 3, 'critical': 3, 'emergency': 3, 'immediately': 3,
    'asap': 3, 'outage': 3, 'breach': 3, 'down': 3, 'failure': 3,
    'dringend': 3, 'kritisch': 3, 'notfall': 3, 'sofort': 3,
    'ausfall': 3, 'angriff': 3,
    'slow': 2, 'error': 2, 'broken': 2, 'failing': 2, 'crash': 2,
    'langsam': 2, 'fehler': 2, 'absturz': 2,
    'request': 1, 'question': 1, 'update': 1, 'inquiry': 1,
    'anfrage': 1, 'frage': 1,
}

def urgency_score(text):
    if not isinstance(text, str): return 0
    text = text.lower()
    return sum(w for kw, w in URGENCY_WORDS.items() if kw in text)

def clean_text(text):
    if not isinstance(text, str): return ''
    text = re.sub(r'\\n', ' ', text)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_priority(subject, body, ticket_type='Incident', queue='Technical Support'):
    """Run prediction and return result dict."""

    # ── Demo mode if model not loaded ────────────────────────────────────────
    if not model_loaded:
        text_lower = (subject + ' ' + body).lower()
        if any(w in text_lower for w in ['urgent','critical','breach','outage','emergency','down','crash']):
            pred, probs = 2, [0.08, 0.12, 0.80]
        elif any(w in text_lower for w in ['question','billing','inquiry','request','info','how to']):
            pred, probs = 0, [0.82, 0.13, 0.05]
        else:
            pred, probs = 1, [0.15, 0.72, 0.13]
        labels = {0: 'Low', 1: 'Medium', 2: 'High'}
        return {
            'priority': labels[pred],
            'confidence': round(max(probs) * 100, 1),
            'probabilities': {
                'low':    round(probs[0] * 100, 1),
                'medium': round(probs[1] * 100, 1),
                'high':   round(probs[2] * 100, 1),
            },
            'demo_mode': True
        }

    # ── Real prediction ───────────────────────────────────────────────────────
    text     = subject + ' ' + subject + ' ' + body
    text     = clean_text(text)
    tfidf_vec = tfidf.transform([text])

    urg   = urgency_score(subject + ' ' + body)
    s_len = len(subject)
    b_len = len(body)
    w_cnt = len(text.split())
    is_rq = 1 if ticket_type.lower() == 'request' else 0
    t_enc = le_type.transform([ticket_type])[0]  if ticket_type in le_type.classes_  else 0
    q_enc = le_queue.transform([queue])[0]        if queue        in le_queue.classes_ else 0

    num_vec = csr_matrix([[s_len, b_len, w_cnt, urg, is_rq, t_enc, q_enc]])
    X_new   = hstack([tfidf_vec, num_vec])

    pred  = model.predict(X_new)[0]
    probs = model.predict_proba(X_new)[0].tolist()

    labels = {0: 'Low', 1: 'Medium', 2: 'High'}
    return {
        'priority':    labels[pred],
        'confidence':  round(max(probs) * 100, 1),
        'probabilities': {
            'low':    round(probs[0] * 100, 1),
            'medium': round(probs[1] * 100, 1),
            'high':   round(probs[2] * 100, 1),
        },
        'demo_mode': False
    }

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html', model_loaded=model_loaded)

@app.route('/predict', methods=['POST'])
def predict():
    data        = request.get_json()
    subject     = data.get('subject', '').strip()
    body        = data.get('body', '').strip()
    ticket_type = data.get('ticket_type', 'Incident')
    queue       = data.get('queue', 'Technical Support')

    if not subject and not body:
        return jsonify({'error': 'Please enter a ticket subject or description.'}), 400

    result = predict_priority(subject, body, ticket_type, queue)
    return jsonify(result)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\n🎫 IT Ticket Priority Classifier")
    print(f"   Running on port {port}\n")
    app.run(host='0.0.0.0', port=port, debug=False)
