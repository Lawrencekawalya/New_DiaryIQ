from flask import Flask, render_template, request, redirect, url_for, session
import firebase_admin
from firebase_admin import credentials
import joblib
import pandas as pd
from datetime import timedelta
import os

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'change_this_to_something_secret')
app.permanent_session_lifetime = timedelta(minutes=30)

# Initialize Firebase
cred = credentials.Certificate("firebase/diaryiq-firebase-adminsdk-fbsvc-4465f48c80.json")
firebase_admin.initialize_app(cred)

# Load model
model = joblib.load("ml_model/dairy_model_3class.pkl")
labels = ['Low', 'Moderate', 'High']

# Normal ranges
NORMAL_RANGES = {
    'pH':                   (6.6,   6.8),
    'Temperature':         (None,  4),
    'Fat_Content':         (3.25, None),
    'SNF':                  (8.25, None),
    'Titratable_Acidity':   (0.13, 0.17),
    'Protein_Content':      (3.0,   3.5),
    'Lactose_Content':      (4.5,   5.2),
    'TPC':                  (None, 100000),
    'SCC':                  (None, 400000),
}

@app.route('/')
def login_page():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    email = request.form['username']
    password = request.form['password']
    if email and password:
        session.permanent = True
        session['user'] = email
        return redirect(url_for('index'))
    return render_template('login.html', error="Invalid credentials")

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login_page'))

@app.route('/index')
def index():
    if 'user' not in session:
        return redirect(url_for('login_page'))
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'user' not in session:
        return redirect(url_for('login_page'))

    # 1) Collect input
    raw = {
        'pH':                 float(request.form['ph']),
        'Temperature':        float(request.form['temperature']),
        'Fat_Content':        float(request.form['fat']),
        'SNF':                float(request.form['snf']),
        'Titratable_Acidity': float(request.form['acidity']),
        'Protein_Content':    float(request.form['protein']),
        'Lactose_Content':    float(request.form['lactose']),
        'TPC':                float(request.form['tpc']),
        'SCC':                float(request.form['scc']),
    }

    # 2) Predict
    df = pd.DataFrame([list(raw.values())], columns=list(raw.keys()))
    pred_idx = model.predict(df)[0]
    prediction = labels[pred_idx]

    # 3) Build color list
    colors = []
    for feat, val in raw.items():
        low, high = NORMAL_RANGES[feat]
        is_normal = True
        if low is not None and val < low:
            is_normal = False
        if high is not None and val > high:
            is_normal = False
        colors.append('#2ecc71' if is_normal else '#e67e22')

        return render_template('result.html',
        prediction=prediction,
        feature_names=list(raw.keys()),
        feature_values=list(raw.values()),  # Actual values for bar chart
        raw_values=list(raw.values()),      # Same as feature_values
        colors=colors
    )

if __name__ == '__main__':
    app.run(debug=True)
