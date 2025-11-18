from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained ML model and label encoder
model = joblib.load("Best_Crop_Recommendation_Model.pkl")
label_encoder = joblib.load("Crop_Label_Encoder.pkl")

@app.route('/')
def welcome():
    return render_template("welcome.html")   # Starting page

@app.route('/index')
def index():
    return render_template("index.html")     # Prediction form page

@app.route('/predict', methods=['POST'])
def predict():
    # Collect form input
    N = float(request.form['N'])
    P = float(request.form['P'])
    K = float(request.form['K'])
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    ph = float(request.form['ph'])
    rainfall = float(request.form['rainfall'])

    # Prepare input for model
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = model.predict(features)[0]

    crop = label_encoder.inverse_transform([prediction])[0]

    # Image file name (auto detects from crop name)
    image_file = crop.lower() + ".jpg"

    return render_template(
        "result.html",
        crop=crop,
        image_file=image_file
    )


if __name__ == "__main__":
    app.run(debug=True)
