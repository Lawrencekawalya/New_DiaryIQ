# from flask import Flask, render_template, request, redirect, url_for, session
# import firebase_admin
# from firebase_admin import credentials
# import joblib
# import pandas as pd
# import numpy as np
# from datetime import timedelta
# import os

# # Initialize Flask app
# app = Flask(__name__)

# # Secret key for session management
# app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'e3cbe607bba16d81f75ae0a7b2a3dbb3e202dbb7114ebea6169ac568f9ed9e3f')
# app.permanent_session_lifetime = timedelta(minutes=30)

# # Initialize Firebase Admin SDK
# cred = credentials.Certificate("firebase/diaryiq-firebase-adminsdk-fbsvc-4465f48c80.json")
# firebase_admin.initialize_app(cred)

# # Load ML model
# model = joblib.load("ml_model/dairy_model_3class.pkl")
# labels = ['Low', 'Moderate', 'High']

# # Route: Login page
# @app.route('/')
# def login_page():
#     return render_template('login.html')

# # Route: Handle login form
# @app.route('/login', methods=['POST'])
# def login():
#     email = request.form['username']
#     password = request.form['password']

#     # NOTE: Firebase Admin SDK does not support user login validation.
#     # Use frontend SDK for real authentication, here we simulate login.
#     if email and password:
#         session.permanent = True
#         session['user'] = email
#         return redirect(url_for('index'))
#     else:
#         return render_template('login.html', error="Invalid credentials")

# # Route: Logout
# @app.route('/logout')
# def logout():
#     session.pop('user', None)
#     return redirect(url_for('login_page'))

# # Route: Prediction form (protected)
# @app.route('/index.html')
# def index():
#     if 'user' not in session:
#         return redirect(url_for('login_page'))
#     return render_template('index.html')

# # Route: Prediction logic (protected)
# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'user' not in session:
#         return redirect(url_for('login_page'))

#     try:
#         # Extract input values
#         pH = float(request.form['ph'])
#         temperature = float(request.form['temperature'])
#         fat = float(request.form['fat'])
#         snf = float(request.form['snf'])
#         acidity = float(request.form['acidity'])
#         protein = float(request.form['protein'])
#         lactose = float(request.form['lactose'])
#         tpc = float(request.form['tpc'])
#         scc = float(request.form['scc'])

#         # Create input DataFrame
#         columns = ['pH', 'Temperature', 'Fat_Content', 'SNF', 'Titratable_Acidity',
#                    'Protein_Content', 'Lactose_Content', 'TPC', 'SCC']
#         input_data = pd.DataFrame([[pH, temperature, fat, snf, acidity, protein, lactose, tpc, scc]],
#                                   columns=columns)

#         prediction = model.predict(input_data)[0]
#         result = labels[prediction]

#         return render_template('index.html', prediction=result)

#     except Exception as e:
#         return f"Error: {str(e)}"

# # Run the app
# if __name__ == '__main__':
#     app.run(debug=True)

# from flask import (
#     Flask, render_template, request,
#     redirect, url_for, session
# )
# import firebase_admin
# from firebase_admin import credentials
# import joblib
# import pandas as pd
# import numpy as np
# from datetime import timedelta
# import os

# # ─── Flask / Session Setup ─────────────────────────────────────────────────────
# app = Flask(__name__)
# app.secret_key = os.environ.get(
#     'FLASK_SECRET_KEY',
#     'e3cbe607bba16d81f75ae0a7b2a3dbb3e202dbb7114ebea6169ac568f9ed9e3f'
# )
# app.permanent_session_lifetime = timedelta(minutes=30)

# # ─── Firebase Admin SDK (service account) ──────────────────────────────────────
# cred = credentials.Certificate(
#     "firebase/diaryiq-firebase-adminsdk-fbsvc-4465f48c80.json"
# )
# firebase_admin.initialize_app(cred)

# # ─── Load your 3-class RandomForest model & labels ─────────────────────────────
# model  = joblib.load("ml_model/dairy_model_3class.pkl")
# labels = ['Low', 'Moderate', 'High']

# # ─── Define your “normal” ranges ────────────────────────────────────────────────
# normal_ranges = {
#     'pH':                   (6.6,    6.8),
#     'Temperature':         (None,   4.0),    # ≤4°C
#     'Fat_Content':         (3.25,   None),
#     'SNF':                 (8.25,   None),
#     'Titratable_Acidity':  (0.13,   0.17),
#     'Protein_Content':     (3.0,    3.5),
#     'Lactose_Content':     (4.5,    5.2),
#     'TPC':                 (None,   100000),  # CFU/ml
#     'SCC':                 (None,   400000)   # cells/ml
# }

# # ─── Routes ───────────────────────────────────────────────────────────────────

# @app.route('/')
# def login_page():
#     return render_template('login.html')


# @app.route('/login', methods=['POST'])
# def login():
#     email    = request.form['username']
#     password = request.form['password']

#     # NOTE: for real auth wire up your frontend Firebase SDK.
#     if email and password:
#         session.permanent = True
#         session['user']  = email
#         return redirect(url_for('index'))
#     return render_template('login.html', error="Invalid credentials")


# @app.route('/logout')
# def logout():
#     session.clear()
#     return redirect(url_for('login_page'))


# @app.route('/index.html')
# def index():
#     if 'user' not in session:
#         return redirect(url_for('login_page'))
#     return render_template('index.html')


# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'user' not in session:
#         return redirect(url_for('login_page'))

#     # ─── 1) Read & parse form inputs ───────────────────────────────────────────
#     raw = {
#         'pH':                   float(request.form['ph']),
#         'Temperature':         float(request.form['temperature']),
#         'Fat_Content':         float(request.form['fat']),
#         'SNF':                 float(request.form['snf']),
#         'Titratable_Acidity':  float(request.form['acidity']),
#         'Protein_Content':     float(request.form['protein']),
#         'Lactose_Content':     float(request.form['lactose']),
#         'TPC':                 float(request.form['tpc']),
#         'SCC':                 float(request.form['scc']),
#     }

#     # ─── 2) Build DataFrame & predict ───────────────────────────────────────────
#     df = pd.DataFrame([list(raw.values())], columns=list(raw.keys()))
#     pred_index = model.predict(df)[0]
#     prediction = labels[pred_index]

#     # ─── 3) Color each slice green/orange by checking normal_ranges ────────────
#     colors = []
#     for feature, value in raw.items():
#         lo, hi = normal_ranges[feature]
#         ok = True
#         if lo is not None and value < lo: ok = False
#         if hi is not None and value > hi: ok = False
#         colors.append('#27ae60' if ok else '#e67e22')

#     # ─── 4) Render the chart + result ──────────────────────────────────────────
#     return render_template(
#         'result.html',
#         inputs=raw,
#         colors=colors,
#         prediction=prediction
#     )


# # ─── Main ─────────────────────────────────────────────────────────────────────
# if __name__ == '__main__':
#     app.run(debug=True)


# from flask import Flask, render_template, request, redirect, url_for, session
# import firebase_admin
# from firebase_admin import credentials
# import joblib
# import pandas as pd
# import numpy as np
# from datetime import timedelta
# import os

# # ─── Setup ─────────────────────────────────────────────────────────────────────
# app = Flask(__name__)
# app.secret_key = os.environ.get('FLASK_SECRET_KEY',
#     'e3cbe607bba16d81f75ae0a7b2a3dbb3e202dbb7114ebea6169ac568f9ed9e3f')
# app.permanent_session_lifetime = timedelta(minutes=30)

# # init Firebase Admin SDK (for anything you need later)
# cred = credentials.Certificate("firebase/diaryiq-firebase-adminsdk-fbsvc-4465f48c80.json")
# firebase_admin.initialize_app(cred)

# # load your 3-class RF model
# model  = joblib.load("ml_model/dairy_model_3class.pkl")
# labels = ['Low', 'Moderate', 'High']

# # define “normal” bounds for each feature
# normal_ranges = {
#     'pH':                (6.6,   6.8),
#     'Temperature':       (None,  4),
#     'Fat_Content':       (3.25,  None),
#     'SNF':               (8.25,  None),
#     'Titratable_Acidity':(0.13,  0.17),
#     'Protein_Content':   (3.0,   3.5),
#     'Lactose_Content':   (4.5,   5.2),
#     'TPC':               (None, 100000),
#     'SCC':               (None, 400000),
# }

# # ─── Authentication Pages ─────────────────────────────────────────────────────
# @app.route('/')
# def login_page():
#     return render_template('login.html')

# @app.route('/login', methods=['POST'])
# def login():
#     email    = request.form['username']
#     password = request.form['password']
#     # (You’d normally verify against Firebase or your DB here)
#     if email and password:
#         session.permanent = True
#         session['user']   = email
#         return redirect(url_for('index'))
#     return render_template('login.html', error="Invalid credentials")

# @app.route('/logout')
# def logout():
#     session.pop('user', None)
#     return redirect(url_for('login_page'))

# # ─── Data Input Form ──────────────────────────────────────────────────────────
# @app.route('/index.html')
# def index():
#     if 'user' not in session:
#         return redirect(url_for('login_page'))
#     return render_template('index.html')

# # ─── Prediction & Chart Data ──────────────────────────────────────────────────
# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'user' not in session:
#         return redirect(url_for('login_page'))

#     # pull form values
#     pH           = float(request.form['ph'])
#     temperature  = float(request.form['temperature'])
#     fat          = float(request.form['fat'])
#     snf          = float(request.form['snf'])
#     acidity      = float(request.form['acidity'])
#     protein      = float(request.form['protein'])
#     lactose      = float(request.form['lactose'])
#     tpc          = float(request.form['tpc'])
#     scc          = float(request.form['scc'])

#     # raw dict to preserve order
#     raw = {
#         'pH': pH,
#         'Temperature': temperature,
#         'Fat_Content': fat,
#         'SNF': snf,
#         'Titratable_Acidity': acidity,
#         'Protein_Content': protein,
#         'Lactose_Content': lactose,
#         'TPC': tpc,
#         'SCC': scc
#     }

#     # run the model
#     df         = pd.DataFrame([list(raw.values())], columns=list(raw.keys()))
#     pred_index = model.predict(df)[0]
#     result     = labels[pred_index]

#     # assign colors (green if within normal_ranges, else orange)
#     colors = []
#     for feat, val in raw.items():
#         lo, hi = normal_ranges[feat]
#         ok = True
#         if lo is not None and val < lo: ok = False
#         if hi is not None and val > hi: ok = False
#         colors.append('#28a745' if ok else '#fd7e14')

#     # pass to template
#     return render_template('result.html',
#         feature_names  = list(raw.keys()),
#         feature_values = list(raw.values()),
#         colors         = colors,
#         prediction     = result
#     )

# if __name__ == '__main__':
#     app.run(debug=True)


# from flask import Flask, render_template, request, redirect, url_for, session
# import firebase_admin
# from firebase_admin import credentials
# import joblib
# import pandas as pd
# import numpy as np
# from datetime import timedelta
# import os

# app = Flask(__name__)

# # session + secret key
# app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'change_this_to_something_secret')
# app.permanent_session_lifetime = timedelta(minutes=30)

# # Firebase Admin init
# cred = credentials.Certificate("firebase/diaryiq-firebase-adminsdk-fbsvc-4465f48c80.json")
# firebase_admin.initialize_app(cred)

# # Load ML model + labels
# model = joblib.load("ml_model/dairy_model_3class.pkl")
# labels = ['Low', 'Moderate', 'High']

# # Define normal ranges for each feature
# NORMAL_RANGES = {
#     'pH':                   (6.6,   6.8),
#     'Temperature':         (None,  4),
#     'Fat_Content':          (3.25, None),
#     'SNF':                  (8.25, None),
#     'Titratable_Acidity':   (0.13, 0.17),
#     'Protein_Content':      (3.0,   3.5),
#     'Lactose_Content':      (4.5,   5.2),
#     'TPC':                  (None,100000),
#     'SCC':                  (None,400000),
# }

# @app.route('/')
# def login_page():
#     return render_template('login.html')

# @app.route('/login', methods=['POST'])
# def login():
#     email = request.form['username']
#     password = request.form['password']
#     # NOTE: Replace with real Firebase auth on frontend
#     if email and password:
#         session.permanent = True
#         session['user'] = email
#         return redirect(url_for('index'))
#     return render_template('login.html', error="Invalid credentials")

# @app.route('/logout')
# def logout():
#     session.pop('user', None)
#     return redirect(url_for('login_page'))

# @app.route('/index')
# def index():
#     if 'user' not in session:
#         return redirect(url_for('login_page'))
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'user' not in session:
#         return redirect(url_for('login_page'))

#     # 1) Grab form inputs
#     raw = {
#         'pH':                 float(request.form['ph']),
#         'Temperature':       float(request.form['temperature']),
#         'Fat_Content':        float(request.form['fat']),
#         'SNF':                float(request.form['snf']),
#         'Titratable_Acidity': float(request.form['acidity']),
#         'Protein_Content':    float(request.form['protein']),
#         'Lactose_Content':    float(request.form['lactose']),
#         'TPC':                float(request.form['tpc']),
#         'SCC':                float(request.form['scc']),
#     }

#     # 2) Predict
#     df = pd.DataFrame([list(raw.values())], columns=list(raw.keys()))
#     pred_idx = model.predict(df)[0]
#     prediction = labels[pred_idx]

#     # 3) Build chart arrays
#     feature_names  = list(raw.keys())
#     feature_values = list(raw.values())
#     colors = []
#     for feat, val in raw.items():
#         low, high = NORMAL_RANGES[feat]
#         ok = True
#         if low is not None  and val < low:  ok = False
#         if high is not None and val > high: ok = False
#         colors.append('#2ecc71' if ok else '#e67e22')

#     # 4) Render results
#     return render_template('result.html',
#         prediction=prediction,
#         feature_names=feature_names,
#         feature_values=feature_values,
#         colors=colors
#     )

# if __name__ == '__main__':
#     app.run(debug=True)

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

    # 3) Normalize values for chart display
    raw_values = list(raw.values())
    min_val = min(raw_values)
    max_val = max(raw_values)
    normalized_values = [(v - min_val) / (max_val - min_val) if max_val != min_val else 1 for v in raw_values]

    # 4) Build color list
    colors = []
    for feat, val in raw.items():
        low, high = NORMAL_RANGES[feat]
        is_normal = True
        if low is not None and val < low:
            is_normal = False
        if high is not None and val > high:
            is_normal = False
        colors.append('#2ecc71' if is_normal else '#e67e22')

    # 5) Pass everything to result template
    return render_template('result.html',
        prediction=prediction,
        feature_names=list(raw.keys()),
        feature_values=normalized_values,
        raw_values=raw_values,
        colors=colors
    )

if __name__ == '__main__':
    app.run(debug=True)
