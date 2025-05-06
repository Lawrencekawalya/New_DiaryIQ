from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("ml_model/dairy_model.pkl")

@app.route('/')
def index():
    return render_template('index.html')  # Will create this HTML next

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        pH = float(request.form['ph'])
        temperature = float(request.form['temperature'])
        fat = float(request.form['fat'])
        conductivity = float(request.form['conductivity'])
        density = float(request.form['density'])

        # Prepare input
        features = np.array([[pH, temperature, fat, conductivity, density]])

        # Predict
        prediction = model.predict(features)[0]
        result = "High" if prediction == 1 else "Low"

        return render_template('index.html', prediction=result)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
