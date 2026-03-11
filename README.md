# 🎫 IT Ticket Priority Classifier — Web App

A clean web application that classifies IT support tickets as **Low / Medium / High** priority using your saved Random Forest model.

---

## 🚀 Setup & Run

### Step 1 — Place your saved model in this folder
Copy your saved model file into the `ticket_app/` folder:
```
ticket_app/
├── app.py
├── requirements.txt
├── README.md
├── ticket_priority_FINAL.joblib   ← put it here
└── templates/
    └── index.html
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Run the app
```bash
python app.py
```

### Step 4 — Open in browser
```
http://localhost:5000
```

---

## 💡 Features
- Paste any ticket subject + description and get instant priority prediction
- Confidence score and probability bars for all three classes
- 5 pre-loaded example tickets to try
- Works with or without the model (demo mode if model not found)
- Supports ticket type and support queue metadata

---

## 📁 Model File
The app expects: `ticket_priority_FINAL.joblib`

This file must contain:
- `model`    — trained Random Forest classifier
- `tfidf`    — fitted TF-IDF vectoriser
- `le_type`  — LabelEncoder for ticket type
- `le_queue` — LabelEncoder for support queue

If the model file is not found, the app runs in **demo mode** using keyword rules.

---

## 🔧 Tech Stack
- Python + Flask (backend)
- Vanilla HTML/CSS/JS (frontend)
- scikit-learn + joblib (ML)
