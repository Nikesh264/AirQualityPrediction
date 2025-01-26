from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)

# Define optimal ranges for pollutants
optimal_ranges = {
    'PM2.5': {'optimal': 35, 'current': None},
    'PM10': {'optimal': 50, 'current': None},
    'NO2': {'optimal': 40, 'current': None},
    'SO2': {'optimal': 40, 'current': None},
    'CO': {'optimal': 0.9, 'current': None},
    'O3': {'optimal': 180, 'current': None}
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    city = request.form['city']
    
    pm25 = float(request.form['pm25'])
    pm10 = float(request.form['pm10'])
    no2 = float(request.form['no2'])
    so2 = float(request.form['so2'])
    co = float(request.form['co'])
    o3 = float(request.form['o3'])

    # Store current values in the optimal_ranges dictionary
    optimal_ranges['PM2.5']['current'] = pm25
    optimal_ranges['PM10']['current'] = pm10
    optimal_ranges['NO2']['current'] = no2
    optimal_ranges['SO2']['current'] = so2
    optimal_ranges['CO']['current'] = co
    optimal_ranges['O3']['current'] = o3

    user_features = np.array([[pm25, pm10, no2, so2, co, o3]])
    
    # Load the appropriate scaler and model based on selected city.
    scaler_path = f'models/{city.lower()}_scaler.pkl'
    
    if not os.path.exists(scaler_path):
        return f"Error: Scaler file does not exist: {scaler_path}. Please ensure that the model files exist."

    scaler = joblib.load(scaler_path)
    
    user_features_scaled = scaler.transform(user_features)

    rf_model = joblib.load(f'models/{city.lower()}_model.pkl')
    xgb_model = joblib.load(f'models/{city.lower()}_xgb_model.pkl')
    gb_model = joblib.load(f'models/{city.lower()}_gb_model.pkl')
    svm_model = joblib.load(f'models/{city.lower()}_svm_model.pkl')

    predicted_aqi_rf = rf_model.predict(user_features_scaled)[0]
    predicted_aqi_xgb = xgb_model.predict(user_features_scaled)[0]
    predicted_aqi_gb = gb_model.predict(user_features_scaled)[0]
    predicted_aqi_svm = svm_model.predict(user_features_scaled)[0]

    avg_aqi_prediction = np.mean([predicted_aqi_rf, predicted_aqi_xgb, predicted_aqi_gb, predicted_aqi_svm])
    
    final_category = categorize_aqi(avg_aqi_prediction)
    
    recommendations = generate_recommendations(final_category)

    # Generate pollutant impact analysis (Cell-17 output)
    pollutant_analysis = generate_pollutant_analysis()

    # Combine both recommendations
    final_recommendations = f"{recommendations} <br><br> {pollutant_analysis}"

    return render_template('result.html', avg_aqi=avg_aqi_prediction, category=final_category, recommendations=final_recommendations)

def categorize_aqi(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"

def generate_recommendations(category):
    recommendations_dict = {
        "Good": "Air quality is safe. Enjoy outdoor activities!",
        "Moderate": "Air quality is acceptable. Sensitive individuals should avoid prolonged outdoor activities.",
        "Unhealthy for Sensitive Groups": "Sensitive groups should reduce outdoor exposure.",
        "Unhealthy": "Everyone should reduce prolonged outdoor exertion.",
        "Very Unhealthy": "Health alert: Avoid outdoor activities.",
        "Hazardous": "Emergency: Stay indoors."
    }
    
    return recommendations_dict.get(category)

def generate_pollutant_analysis():
    results = []
    for pollutant, values in optimal_ranges.items():
        current_level = values['current']
        optimal_level = values['optimal']
        print(f"Checking {pollutant}: Current level is {current_level}, Optimal level is {optimal_level}")  # Debugging line

        if current_level > optimal_level:
            reduction_needed = current_level - optimal_level
            results.append(
                f"<strong>{pollutant}:</strong> Exceeds optimal range<br>"
                f"Current: {current_level} µg/m³ | Optimal: ≤{optimal_level} µg/m³<br>"
                f"* Reduce {pollutant} by {reduction_needed:.2f} µg/m³.<br>"
                f"* Suggestions:<br>"
                f"- Implement measures to reduce emissions from vehicles and industries.<br>"
            )
        else:
            results.append(
                f"<strong>{pollutant}:</strong> Within optimal range<br>"
                f"Current: {current_level} µg/m³ | Optimal: ≤{optimal_level} µg/m³<br>"
            )
    
    return "<br>".join(results)

if __name__ == '__main__':
    app.run(debug=True)
