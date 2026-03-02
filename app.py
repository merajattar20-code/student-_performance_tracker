from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [
        float(request.form['study_hours']),
        float(request.form['attendance']),
        float(request.form['previous_marks']),
        float(request.form['sleep_hours']),
        float(request.form['internet_usage'])
    ]
    
    prediction = model.predict([features])
    
    result = prediction[0]
    
    return render_template('index.html', prediction_text=f'Predicted Performance: {result}')

if __name__ == '__main__':
    app.run(debug=True)
