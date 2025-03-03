from flask import Flask, render_template, request, redirect, url_for
from joblib import load
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

app = Flask(__name__)

# Load the model, scaler, and polynomial transformer
model = load(r"C:\flaskenv\svr_cancer_model_optimized_v3.pkl")
scaler = load(r"C:\flaskenv\cc_scaler.pkl")
poly = load(r"C:\flaskenv\cc_poly.pkl")

# Initialize label encoders for categorical variables
gender_encoder = LabelEncoder()
smoking_encoder = LabelEncoder()
cancerhistory_encoder = LabelEncoder()

# Example training categories (adjust to match your training data)
gender_encoder.fit(['Male', 'Female'])
smoking_encoder.fit(['Non-Smoker', 'Smoker'])
cancerhistory_encoder.fit(['No', 'Yes'])

# CSV file path to store prediction history
history_file = r"C:\Users\shalu\Downloads\prediction_history.csv"

# Ensure the CSV file exists and create it with headers if not
if not os.path.exists(history_file):
    pd.DataFrame(columns=[
        'Age', 'Gender', 'BMI', 'Smoking', 'Genetic Risk',
        'Physical Activity', 'Alcohol Intake', 'Cancer History', 
        'Prediction Score', 'Risk Level'
    ]).to_csv(history_file, index=False)

@app.route('/')
def home():
    return render_template('ascc.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from the form
        age = float(request.form['age'])
        gender = request.form['gender']
        bmi = float(request.form['bmi'])
        smoking = request.form['smoking']
        geneticrisk = float(request.form['geneticrisk'])
        physical_activity = float(request.form['physical_activity'])
        alcohol_intake = float(request.form['alcohol_intake'])
        cancerhistory = request.form['cancerhistory']

        # Encode categorical features
        gender_encoded = gender_encoder.transform([gender])[0]
        smoking_encoded = smoking_encoder.transform([smoking])[0]
        cancerhistory_encoded = cancerhistory_encoder.transform([cancerhistory])[0]

        # Prepare input data
        input_data = np.array([[age, gender_encoded, bmi, smoking_encoded, geneticrisk, physical_activity, alcohol_intake, cancerhistory_encoded]])
        input_data_poly = poly.transform(input_data)
        input_data_scaled = scaler.transform(input_data_poly)

        # Make a prediction
        prediction = model.predict(input_data_scaled)
        prediction_value = prediction[0]

        # Classify risk levels
        if prediction_value < 0.5:
            diagnosis_text = 'Low Risk of Cancer'
        elif 0.5 <= prediction_value < 0.85:
            diagnosis_text = 'Moderate Risk of Cancer'
        else:
            diagnosis_text = 'High Risk of Cancer'

        # Save prediction to history
        new_record = {
            'Age': age, 'Gender': gender, 'BMI': bmi, 'Smoking': smoking,
            'Genetic Risk': geneticrisk, 'Physical Activity': physical_activity,
            'Alcohol Intake': alcohol_intake, 'Cancer History': cancerhistory,
            'Prediction Score': round(prediction_value, 2), 'Risk Level': diagnosis_text
        }
        pd.DataFrame([new_record]).to_csv(history_file, mode='a', header=False, index=False)

        # Redirect to the prediction page with results
        return redirect(url_for('pred', prediction_text=f'Predicted Diagnosis: {diagnosis_text} (Score: {prediction_value:.2f})'))

    except Exception as e:
        return render_template('ascc.html', prediction_text=f'Error: {str(e)}')

@app.route('/pred')
def pred():
    prediction_text = request.args.get('prediction_text')
    return render_template('pred.html', prediction_text=prediction_text)

@app.route('/history')
def view_history():
    try:
        # Load prediction history
        history_df = pd.read_csv(history_file)

        # Clean column names to avoid unexpected spaces
        history_df.columns = history_df.columns.str.strip()

        # Pass HTML table string directly to avoid list artifacts
        return render_template('history.html', tables=history_df.to_html(classes='table table-bordered table-striped', index=False))
    except Exception as e:
        return render_template('history.html', error=f"Error loading history: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
