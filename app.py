from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load the trained model
model_path = 'diabetes_prediction_model.pkl'
model_info = joblib.load(model_path)
model = model_info['model']
feature_names = model_info['feature_names']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        try:
            # Extract basic features from form
            
            pregnancies = float(request.form['pregnancies'])
            glucose = float(request.form['glucose'])
            blood_pressure = float(request.form['bloodpressure'])
            skin_thickness = float(request.form['skinthickness'])
            insulin = float(request.form['insulin'])
            bmi = float(request.form['bmi'])
            diabetes_pedigree = float(request.form['diabetespedigree'])
            age = float(request.form['age'])
            
            # Create a DataFrame with the input values
            input_data = pd.DataFrame({
                'Pregnancies': [pregnancies],
                'Glucose': [glucose],
                'BloodPressure': [blood_pressure],
                'SkinThickness': [skin_thickness],
                'Insulin': [insulin],
                'BMI': [bmi],
                'DiabetesPedigreeFunction': [diabetes_pedigree],
                'Age': [age]
            })
            
            # Feature engineering (same as in the training code)
            # BMI Categories
            input_data['BMI_Category'] = pd.cut(
                input_data['BMI'],
                bins=[0, 18.5, 25, 30, 100],
                labels=['Underweight', 'Normal', 'Overweight', 'Obese']
            )

            # Age Groups
            input_data['Age_Group'] = pd.cut(
                input_data['Age'],
                bins=[20, 30, 40, 50, 100],
                labels=['20-30', '30-40', '40-50', '50+']
            )

            # Glucose Level Categories
            input_data['Glucose_Category'] = pd.cut(
                input_data['Glucose'],
                bins=[0, 70, 100, 126, 200],
                labels=['Low', 'Normal', 'Pre-diabetic', 'Diabetic']
            )

            # Create interaction features
            input_data['Glucose_BMI'] = input_data['Glucose'] * input_data['BMI']
            input_data['Age_Pregnancies'] = input_data['Age'] * input_data['Pregnancies']

            # Convert categorical features to dummy variables
            input_data = pd.get_dummies(
                input_data, 
                columns=['BMI_Category', 'Age_Group', 'Glucose_Category'],
                drop_first=True
            )
            
            # Ensure all columns from training are present
            for feature in feature_names:
                if feature not in input_data.columns:
                    input_data[feature] = 0
            
            # Reorder columns to match training data
            input_data = input_data[feature_names]
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            prediction_proba = model.predict_proba(input_data)[0][1]
            
            # Return result
            result = {
                'prediction': int(prediction),
                'probability': float(prediction_proba),
                'risk_level': 'High' if prediction_proba > 0.7 else 'Medium' if prediction_proba > 0.4 else 'Low'
            }
            
            return render_template('result.html', result=result)
            
        except Exception as e:
            return render_template('error.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)