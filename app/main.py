from flask import Flask, render_template, request, redirect, url_for, session
import firebase_admin
from firebase_admin import credentials
import joblib
import pandas as pd
from datetime import datetime, timedelta
import os
import uuid
import firebase_admin
from firebase_admin import credentials, firestore
from flask import redirect, url_for


app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'change_this_to_something_secret')
app.permanent_session_lifetime = timedelta(minutes=30)

# Initialize Firebase
cred = credentials.Certificate("firebase/diaryiq-firebase-adminsdk-fbsvc-4465f48c80.json")
firebase_admin.initialize_app(cred)

db = firestore.client()

# Load model
model = joblib.load("ml_model/dairy_model_4class.pkl")
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

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'user' not in session:
#         return redirect(url_for('login_page'))

#     # 1) Collect input
#     raw = {
#         'pH':                 float(request.form['ph']),
#         'Temperature':        float(request.form['temperature']),
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
#     # prediction = labels[int(pred_idx)]
#     prediction = pred_idx

#     # 3) Build color list
#     colors = []
#     for feat, val in raw.items():
#         low, high = NORMAL_RANGES[feat]
#         is_normal = True
#         if low is not None and val < low:
#             is_normal = False
#         if high is not None and val > high:
#             is_normal = False
#         colors.append('#2ecc71' if is_normal else '#e67e22')

#     # 4) Build dynamic suggestions (INSERT THIS PART)
#     suggestions = []
#     if raw['Temperature'] > 4:
#         suggestions.append("⚠ Check cooling system – temperature above safe range.")
#     if raw['SCC'] > 400000:
#         suggestions.append("⚠ High SCC – possible mastitis, review herd health.")
#     if raw['Fat_Content'] < 3.25:
#         suggestions.append("⚠ Low fat content – check feed and nutrition.")
#     if raw['TPC'] > 100000:
#         suggestions.append("⚠ High bacterial count – review hygiene and storage.")
#     if raw['pH'] < 6.6 or raw['pH'] > 6.8:
#         suggestions.append("⚠ Abnormal pH – check for contamination or spoilage.")

#     # If no issues found
#     if not suggestions:
#         suggestions.append("✅ Milk meets quality standards.")
#         suggestions.append("✅ Maintain current handling procedures.")

#     # 5) Return result page
#     return render_template(
#         'result.html',
#         prediction=prediction,
#         feature_names=list(raw.keys()),
#         raw_values=list(raw.values()),
#         colors=colors,
#         raw=raw,
#         suggestions=suggestions   # <-- pass suggestions to template
#     )
# ########################################################
# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'user' not in session:
#         return redirect(url_for('login_page'))

#     # 0) Capture Batch Info
#     batch_info = {
#         'Farmer': request.form.get('farmer'),
#         'Contact': request.form.get('contact'),
#         'Location': request.form.get('location'),
#         'Batch Number': request.form.get('batch_number'),
#         'Time of Collection': request.form.get('time_of_collection'),
#         'Transport Details': request.form.get('transport_details')
#     }

#     # 1) Collect predictor input
#     raw = {
#         'pH':                 float(request.form['ph']),
#         'Temperature':        float(request.form['temperature']),
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
#     prediction = pred_idx

#     # 3) Build color list
#     colors = []
#     for feat, val in raw.items():
#         low, high = NORMAL_RANGES[feat]
#         is_normal = True
#         if low is not None and val < low:
#             is_normal = False
#         if high is not None and val > high:
#             is_normal = False
#         colors.append('#2ecc71' if is_normal else '#e67e22')

#     # 4) Build dynamic suggestions
#     suggestions = []
#     if raw['Temperature'] > 4:
#         suggestions.append("⚠ Check cooling system – temperature above safe range.")
#     if raw['SCC'] > 400000:
#         suggestions.append("⚠ High SCC – possible mastitis, review herd health.")
#     if raw['Fat_Content'] < 3.25:
#         suggestions.append("⚠ Low fat content – check feed and nutrition.")
#     if raw['TPC'] > 100000:
#         suggestions.append("⚠ High bacterial count – review hygiene and storage.")
#     if raw['pH'] < 6.6 or raw['pH'] > 6.8:
#         suggestions.append("⚠ Abnormal pH – check for contamination or spoilage.")

#     if not suggestions:
#         suggestions.append("✅ Milk meets quality standards.")
#         suggestions.append("✅ Maintain current handling procedures.")

#     # 5) Render template with batch_info added
#     return render_template(
#         'result.html',
#         prediction=prediction,
#         feature_names=list(raw.keys()),
#         raw_values=list(raw.values()),
#         colors=colors,
#         raw=raw,
#         suggestions=suggestions,
#         batch_info=batch_info   # <-- pass batch info here
#     )
############################################################################
# @app.route('/history')
# def history():
#     if 'user' not in session:
#         return redirect(url_for('login_page'))

#     # Fetch all batches ordered by timestamp
#     batches = db.collection("milk_batches").order_by("created_at").stream()

#     # Map prediction → number
#     mapping = {"Low": 1, "Moderate": 2, "High": 3}

#     chart_data = []
#     for batch in batches:
#         data = batch.to_dict()
#         created_at = data.get("created_at")

#         # Convert Firestore timestamp to ISO string for Chart.js
#         if created_at:
#             created_at = created_at.isoformat()  

#         chart_data.append({
#             "date": created_at,  # X-axis
#             "prediction": mapping.get(data.get("prediction"), 0)  # Y-axis
#         })

#     return render_template("history.html", chart_data=chart_data)

@app.route('/history')
def history():
    if 'user' not in session:
        return redirect(url_for('login_page'))

    # Fetch all batches ordered by created_at
    batches = db.collection("milk_batches").order_by("created_at").stream()

    history_data = []
    for batch in batches:
        d = batch.to_dict()
        history_data.append({
            "Farmer": d.get("Farmer"),
            "Batch Number": d.get("Batch Number"),
            "Time of Collection": d.get("Time of Collection"),
            "Location": d.get("Location"),
            "Prediction": d.get("prediction"),
        })

    return render_template("history.html", history_data=history_data)



############################################################################
# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'user' not in session:
#         return redirect(url_for('login_page'))

#     # -----------------------------
#     # 0) Capture Batch Info
#     # -----------------------------
#     batch_info = {
#         'Farmer': request.form.get('farmer'),
#         'Contact': request.form.get('contact'),
#         'Location': request.form.get('location'),
#         'Batch Number': request.form.get('batch_number'),
#         'Time of Collection': request.form.get('time_of_collection'),
#         'Transport Details': request.form.get('transport_details')
#     }

#     # -----------------------------
#     # 1) Collect predictor input
#     # -----------------------------
#     raw = {
#         'pH':                 float(request.form['ph']),
#         'Temperature':        float(request.form['temperature']),
#         'Fat_Content':        float(request.form['fat']),
#         'SNF':                float(request.form['snf']),
#         'Titratable_Acidity': float(request.form['acidity']),
#         'Protein_Content':    float(request.form['protein']),
#         'Lactose_Content':    float(request.form['lactose']),
#         'TPC':                float(request.form['tpc']),
#         'SCC':                float(request.form['scc']),
#     }

#     # -----------------------------
#     # 2) Predict
#     # -----------------------------
#     df = pd.DataFrame([list(raw.values())], columns=list(raw.keys()))
#     pred_idx = model.predict(df)[0]
#     prediction = pred_idx

#     # -----------------------------
#     # 3) Build color list
#     # -----------------------------
#     colors = []
#     for feat, val in raw.items():
#         low, high = NORMAL_RANGES[feat]
#         is_normal = True
#         if low is not None and val < low:
#             is_normal = False
#         if high is not None and val > high:
#             is_normal = False
#         colors.append('#2ecc71' if is_normal else '#e67e22')

#     # -----------------------------
#     # 4) Build dynamic suggestions
#     # -----------------------------
#     suggestions = []
#     if raw['Temperature'] > 4:
#         suggestions.append("⚠ Check cooling system – temperature above safe range.")
#     if raw['SCC'] > 400000:
#         suggestions.append("⚠ High SCC – possible mastitis, review herd health.")
#     if raw['Fat_Content'] < 3.25:
#         suggestions.append("⚠ Low fat content – check feed and nutrition.")
#     if raw['TPC'] > 100000:
#         suggestions.append("⚠ High bacterial count – review hygiene and storage.")
#     if raw['pH'] < 6.6 or raw['pH'] > 6.8:
#         suggestions.append("⚠ Abnormal pH – check for contamination or spoilage.")

#     if not suggestions:
#         suggestions.append("✅ Milk meets quality standards.")
#         suggestions.append("✅ Maintain current handling procedures.")

#     # -----------------------------
#     # 5) Save everything to Firestore
#     # -----------------------------
#     batch_doc = {
#         **batch_info,         # farmer, contact, location, batch number, etc.
#         **raw,                # predictor values
#         "prediction": prediction,
#         "created_at": firestore.SERVER_TIMESTAMP
#     }

#     db.collection("milk_batches").add(batch_doc)

#     # -----------------------------
#     # 6) Render template
#     # -----------------------------
#     return render_template(
#         'result.html',
#         prediction=prediction,
#         feature_names=list(raw.keys()),
#         raw_values=list(raw.values()),
#         colors=colors,
#         raw=raw,
#         suggestions=suggestions,
#         batch_info=batch_info   # display current batch in template
#     )

#######################################################################################
# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'user' not in session:
#         return redirect(url_for('login_page'))

#     # -----------------------------
#     # 0) Capture Batch Info
#     # -----------------------------
#     batch_info = {
#         'Farmer': request.form.get('farmer'),
#         'Contact': request.form.get('contact'),
#         'Location': request.form.get('location'),
#         'Batch Number': request.form.get('batch_number'),
#         'Time of Collection': request.form.get('time_of_collection'),
#         'Transport Details': request.form.get('transport_details')
#     }

#     # -----------------------------
#     # 1) Collect predictor input
#     # -----------------------------
#     raw = {
#         'pH':                 float(request.form['ph']),
#         'Temperature':        float(request.form['temperature']),
#         'Fat_Content':        float(request.form['fat']),
#         'SNF':                float(request.form['snf']),
#         'Titratable_Acidity': float(request.form['acidity']),
#         'Protein_Content':    float(request.form['protein']),
#         'Lactose_Content':    float(request.form['lactose']),
#         'TPC':                float(request.form['tpc']),
#         'SCC':                float(request.form['scc']),
#     }

#     # -----------------------------
#     # 2) Predict
#     # -----------------------------
#     df = pd.DataFrame([list(raw.values())], columns=list(raw.keys()))
#     pred_idx = model.predict(df)[0]
#     prediction = pred_idx

#     # -----------------------------
#     # 3) Build color list
#     # -----------------------------
#     colors = []
#     for feat, val in raw.items():
#         low, high = NORMAL_RANGES[feat]
#         is_normal = True
#         if low is not None and val < low:
#             is_normal = False
#         if high is not None and val > high:
#             is_normal = False
#         colors.append('#2ecc71' if is_normal else '#e67e22')

#     # -----------------------------
#     # 4) Build dynamic suggestions
#     # -----------------------------
#     suggestions = []
#     if raw['Temperature'] > 4:
#         suggestions.append("⚠ Check cooling system – temperature above safe range.")
#     if raw['SCC'] > 400000:
#         suggestions.append("⚠ High SCC – possible mastitis, review herd health.")
#     if raw['Fat_Content'] < 3.25:
#         suggestions.append("⚠ Low fat content – check feed and nutrition.")
#     if raw['TPC'] > 100000:
#         suggestions.append("⚠ High bacterial count – review hygiene and storage.")
#     if raw['pH'] < 6.6 or raw['pH'] > 6.8:
#         suggestions.append("⚠ Abnormal pH – check for contamination or spoilage.")

#     if not suggestions:
#         suggestions.append("✅ Milk meets quality standards.")
#         suggestions.append("✅ Maintain current handling procedures.")

#     # -----------------------------
#     # 5) Save everything to Firestore
#     # -----------------------------
#     batch_doc = {
#         **batch_info,
#         **raw,
#         "prediction": prediction,
#         "created_at": firestore.SERVER_TIMESTAMP
#     }
#     db.collection("milk_batches").add(batch_doc)

#     # -----------------------------
#     # 6) Render template
#     # -----------------------------
#     return render_template(
#         'result.html',
#         prediction=prediction,
#         feature_names=list(raw.keys()),
#         raw_values=list(raw.values()),
#         colors=colors,
#         raw=raw,
#         suggestions=suggestions,
#         batch_info=batch_info,
#         chart_data=[]  # empty for current prediction page
#     )

############################final working one###########################################
# QUALITY_MAP = {"Low": 1, "Moderate": 2, "High": 3}
# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'user' not in session:
#         return redirect(url_for('login_page'))
#     # Auto-generate batch number & timestamp
#     batch_info = {
#         'Farmer': request.form.get('farmer'),
#         'Contact': request.form.get('contact'),
#         'Location': request.form.get('location'),
#         'Batch Number': f"BATCH-{uuid.uuid4().hex[:8].upper()}",  # unique ID
#         'Time of Collection': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # readable timestamp
#         'Transport Details': request.form.get('transport_details'),
#     }

#     # 2) Collect predictor inputs
#     raw = {
#         'pH':                 float(request.form['ph']),
#         'Temperature':        float(request.form['temperature']),
#         'Fat_Content':        float(request.form['fat']),
#         'SNF':                float(request.form['snf']),
#         'Titratable_Acidity': float(request.form['acidity']),
#         'Protein_Content':    float(request.form['protein']),
#         'Lactose_Content':    float(request.form['lactose']),
#         'TPC':                float(request.form['tpc']),
#         'SCC':                float(request.form['scc']),
#     }

#     # 3) Run prediction
#     df = pd.DataFrame([list(raw.values())], columns=list(raw.keys()))
#     pred_idx = model.predict(df)[0]
#     prediction = pred_idx

#     # 4) Build color list for result highlighting
#     colors = []
#     for feat, val in raw.items():
#         low, high = NORMAL_RANGES[feat]
#         is_normal = True
#         if low is not None and val < low:
#             is_normal = False
#         if high is not None and val > high:
#             is_normal = False
#         colors.append('#2ecc71' if is_normal else '#e67e22')

#     # 5) Dynamic suggestions
#     suggestions = []
#     if raw['Temperature'] > 4:
#         suggestions.append("⚠ Check cooling system – temperature above safe range.")
#     if raw['SCC'] > 400000:
#         suggestions.append("⚠ High SCC – possible mastitis, review herd health.")
#     if raw['Fat_Content'] < 3.25:
#         suggestions.append("⚠ Low fat content – check feed and nutrition.")
#     if raw['TPC'] > 100000:
#         suggestions.append("⚠ High bacterial count – review hygiene and storage.")
#     if raw['pH'] < 6.6 or raw['pH'] > 6.8:
#         suggestions.append("⚠ Abnormal pH – check for contamination or spoilage.")

#     if not suggestions:
#         suggestions.append("✅ Milk meets quality standards.")
#         suggestions.append("✅ Maintain current handling procedures.")

#     # 6) Save batch to Firestore
#     batch_doc = {**batch_info, **raw, "prediction": prediction, "created_at": firestore.SERVER_TIMESTAMP}
#     db.collection("milk_batches").add(batch_doc)

#     # Fetch all batches ordered by timestamp
#     batches = db.collection("milk_batches").order_by("created_at").stream()

#     mapping = {"Low": 0, "Moderate": 1, "High": 2,".":3}
#     chart_data = []

#     for batch in batches:
#         data = batch.to_dict()
#         created_at = data.get("created_at")
#         if created_at:
#             try:
#                 created_at = created_at.isoformat()  # Firestore timestamp -> ISO string
#             except Exception as e:
#                 print("⚠️ created_at not valid timestamp:", created_at, e)
#                 created_at = None

#         chart_data.append({
#             "date": created_at,
#             "prediction": mapping.get(data.get("prediction"), 0)
#         })

#     # Pass chart_data PLUS empty defaults for the rest
#     return render_template(
#         "result.html",
#         prediction=prediction,
#         feature_names=list(raw.keys()),
#         raw_values=list(raw.values()),
#         colors=colors,
#         raw=raw,
#         suggestions=suggestions,
#         batch_info=batch_info,

#         chart_data=chart_data,
#     )
##############################################################################
# QUALITY_MAP = {"Low": 0, "Moderate": 1, "High": 2}

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'user' not in session:
#         return redirect(url_for('login_page'))

#     # 1) Collect batch info
#     batch_info = {
#         'Farmer': request.form.get('farmer'),
#         'Contact': request.form.get('contact'),
#         'Location': request.form.get('location'),
#         'Batch Number': f"BATCH-{uuid.uuid4().hex[:8].upper()}",
#         'Time of Collection': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#         'Transport Details': request.form.get('transport_details'),
#     }

#     # 2) Collect predictor inputs
#     raw = {
#         'pH':                 float(request.form['ph']),
#         'Temperature':        float(request.form['temperature']),
#         'Fat_Content':        float(request.form['fat']),
#         'SNF':                float(request.form['snf']),
#         'Titratable_Acidity': float(request.form['acidity']),
#         'Protein_Content':    float(request.form['protein']),
#         'Lactose_Content':    float(request.form['lactose']),
#         'TPC':                float(request.form['tpc']),
#         'SCC':                float(request.form['scc']),
#     }

#     # 3) Run prediction
#     df = pd.DataFrame([list(raw.values())], columns=list(raw.keys()))
#     prediction = model.predict(df)[0]   # e.g., "Low", "Moderate", "High"

#     # 4) Build color list
#     colors = []
#     for feat, val in raw.items():
#         low, high = NORMAL_RANGES[feat]
#         is_normal = True
#         if low is not None and val < low:
#             is_normal = False
#         if high is not None and val > high:
#             is_normal = False
#         colors.append('#2ecc71' if is_normal else '#e67e22')

#     # 5) Suggestions
#     suggestions = []
#     if raw['Temperature'] > 4:
#         suggestions.append("⚠ Check cooling system – temperature above safe range.")
#     if raw['SCC'] > 400000:
#         suggestions.append("⚠ High SCC – possible mastitis, review herd health.")
#     if raw['Fat_Content'] < 3.25:
#         suggestions.append("⚠ Low fat content – check feed and nutrition.")
#     if raw['TPC'] > 100000:
#         suggestions.append("⚠ High bacterial count – review hygiene and storage.")
#     if raw['pH'] < 6.6 or raw['pH'] > 6.8:
#         suggestions.append("⚠ Abnormal pH – check for contamination or spoilage.")

#     if not suggestions:
#         suggestions.append("✅ Milk meets quality standards.")
#         suggestions.append("✅ Maintain current handling procedures.")

#     # 6) Save to Firestore
#     batch_doc = {**batch_info, **raw, "prediction": prediction, "created_at": firestore.SERVER_TIMESTAMP}
#     db.collection("milk_batches").add(batch_doc)

#     # 7) Fetch history for chart (using Time of Collection instead of created_at)
#     batches = db.collection("milk_batches").order_by("created_at").stream()
#     chart_data = []
#     for batch in batches:
#         data = batch.to_dict()
#         time_of_collection = data.get("Time of Collection")
#         if not time_of_collection:
#             continue
#         chart_data.append({
#             "date": time_of_collection,  # X-axis
#             "prediction": QUALITY_MAP.get(data.get("prediction"), 0)  # Y-axis numeric
#         })

#     # 8) Render template
#     return render_template(
#         'result.html',
#         prediction=prediction,
#         feature_names=list(raw.keys()),
#         raw_values=list(raw.values()),
#         colors=colors,
#         raw=raw,
#         suggestions=suggestions,
#         batch_info=batch_info,
#         chart_data=chart_data
#     )

QUALITY_MAP = {"Low": 0, "Moderate": 1, "High": 2}

@app.route('/predict', methods=['POST'])
def predict():
    if 'user' not in session:
        return redirect(url_for('login_page'))

    # 1) Collect batch info
    batch_info = {
        'Farmer': request.form.get('farmer'),
        'Contact': request.form.get('contact'),
        'Location': request.form.get('location'),
        'Batch Number': f"BATCH-{uuid.uuid4().hex[:8].upper()}",
        'Time of Collection': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'Transport Details': request.form.get('transport_details'),
    }

    # 2) Collect predictor inputs
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

    # 3) Run prediction
    df = pd.DataFrame([list(raw.values())], columns=list(raw.keys()))
    prediction = model.predict(df)[0]   # "Low", "Moderate", "High"

    # 4) Build color list (for later reuse)
    colors = []
    for feat, val in raw.items():
        low, high = NORMAL_RANGES[feat]
        is_normal = True
        if low is not None and val < low:
            is_normal = False
        if high is not None and val > high:
            is_normal = False
        colors.append('#2ecc71' if is_normal else '#e67e22')

    # 5) Suggestions (for later reuse)
    suggestions = []
    if raw['Temperature'] > 4:
        suggestions.append("⚠ Check cooling system – temperature above safe range.")
    if raw['SCC'] > 400000:
        suggestions.append("⚠ High SCC – possible mastitis, review herd health.")
    if raw['Fat_Content'] < 3.25:
        suggestions.append("⚠ Low fat content – check feed and nutrition.")
    if raw['TPC'] > 100000:
        suggestions.append("⚠ High bacterial count – review hygiene and storage.")
    if raw['pH'] < 6.6 or raw['pH'] > 6.8:
        suggestions.append("⚠ Abnormal pH – check for contamination or spoilage.")

    if not suggestions:
        suggestions.append("✅ Milk meets quality standards.")
        suggestions.append("✅ Maintain current handling procedures.")

    # 6) Save to Firestore
    batch_doc = {
        **batch_info,
        **raw,
        "prediction": prediction,
        "colors": colors,         # store colors for later use
        "suggestions": suggestions,
        "created_at": firestore.SERVER_TIMESTAMP
    }
    doc_ref = db.collection("milk_batches").add(batch_doc)
    batch_id = doc_ref[1].id  # Firestore returns (write_result, doc_ref)

    # 7) Redirect to dedicated result page
    return redirect(url_for('show_result', batch_id=batch_id))

##############################################################
@app.route('/result/<batch_id>')
def show_result(batch_id):
    if 'user' not in session:
        return redirect(url_for('login_page'))

    # Get this batch
    doc = db.collection("milk_batches").document(batch_id).get()
    if not doc.exists:
        return "Batch not found", 404
    data = doc.to_dict()

    # Fetch history for chart
    batches = db.collection("milk_batches").order_by("created_at").stream()
    chart_data = []
    # for batch in batches:
    #     d = batch.to_dict()
    #     if "Time of Collection" not in d:
    #         continue
    #     chart_data.append({
    #         "date": d["Time of Collection"],
    #         "prediction": QUALITY_MAP.get(d.get("prediction"), 0),
    #         "farmer": data.get("Farmer"),
    #         "prediction_label": data.get("prediction")
    #     })
    for batch in batches:
        d = batch.to_dict()
        if "Time of Collection" not in d:
            continue
        chart_data.append({
            "date": d["Time of Collection"],
            "prediction": QUALITY_MAP.get(d.get("prediction"), 0),
            "farmer": d.get("Farmer"),                 # ✅ from this batch
            "prediction_label": d.get("prediction")    # ✅ from this batch
        })
        

    # Render template
    return render_template(
        "result.html",
        prediction=data.get("prediction"),
        feature_names=list(NORMAL_RANGES.keys()),
        raw_values=[data.get(k) for k in NORMAL_RANGES.keys()],
        colors=data.get("colors", []),
        raw={k: data.get(k) for k in NORMAL_RANGES.keys()},
        suggestions=data.get("suggestions", []),
        batch_info={
            "Farmer": data.get("Farmer"),
            "Contact": data.get("Contact"),
            "Location": data.get("Location"),
            "Batch Number": data.get("Batch Number"),
            "Time of Collection": data.get("Time of Collection"),
            "Transport Details": data.get("Transport Details"),
        },
        chart_data=chart_data
    )




if __name__ == '__main__':
    app.run(debug=True)
