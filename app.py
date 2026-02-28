from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model (we will create later)
try:
    model = joblib.load("model.pkl")
except:
    model = None

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return "Model not trained yet!"

    study_hours = float(request.form['study_hours'])
    attendance = float(request.form['attendance'])
    previous_marks = float(request.form['previous_marks'])
    sleep_hours = float(request.form['sleep_hours'])
    internet_usage = float(request.form['internet_usage'])

    input_data = [[study_hours, attendance, previous_marks, sleep_hours, internet_usage]]

    prediction = model.predict(input_data)

    return render_template("index.html", prediction_text=f"Predicted Performance: {prediction[0]}")

if __name__ == "__main__":
    app.run(debug=True)
