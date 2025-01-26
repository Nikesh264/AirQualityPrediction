import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# List of datasets and corresponding model names
datasets = {
    'Warangal': 'datasets/Warangal.csv',
    'Palwancha': 'datasets/Palwancha.csv',
    'Rajendranagar': 'datasets/Rajendranagar.csv'
}

for city, file_path in datasets.items():
    print(f"Training model for {city}...")
    
    try:
        # Load dataset
        data = pd.read_csv(file_path)

        # Preprocess the data
        features = data[['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']]
        target = data['AQI']

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Train models
        rf_model = RandomForestRegressor().fit(X_train_scaled, y_train)
        xgb_model = XGBRegressor(objective='reg:squarederror').fit(X_train_scaled, y_train)
        gb_model = GradientBoostingRegressor().fit(X_train_scaled, y_train)
        svm_model = SVR().fit(X_train_scaled, y_train)

        # Save models and scalers to .pkl files in the models folder based on city name.
        joblib.dump(rf_model, f'models/{city.lower()}_model.pkl')
        joblib.dump(xgb_model, f'models/{city.lower()}_xgb_model.pkl')
        joblib.dump(gb_model, f'models/{city.lower()}_gb_model.pkl')
        joblib.dump(svm_model, f'models/{city.lower()}_svm_model.pkl')

        # Save the scaler specific to each city
        joblib.dump(scaler, f'models/{city.lower()}_scaler.pkl')

        print(f"{city} models and scaler saved successfully!")

    except Exception as e:
        print(f"Error processing {city}: {e}")

print("All models trained and saved successfully!")
