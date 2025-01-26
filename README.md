# Air Quality Prediction

## Project Overview

This project is an **Air Quality Prediction** application developed using Flask and Python. It utilizes machine learning models to predict the Air Quality Index (AQI) based on user-provided pollutant levels (PM2.5, PM10, NO2, SO2, CO, O3). The application aims to provide users with insights into air quality dynamics and recommendations for maintaining good air quality.

## Objectives

- To accurately predict AQI based on input parameters.
- To provide detailed pollutant impact analysis.
- To display general recommendations based on AQI categories.
- To create a user-friendly web interface for easy interaction.

## Features

- Predicts AQI from user input.
- Provides pollutant impact analysis.
- Displays recommendations based on AQI category.
- Built with a responsive and intuitive web interface.

## Technologies Used

- Python: Programming language for backend development.
- Flask: Web framework for building the application.
- Pandas: Data manipulation library.
- NumPy: Library for numerical computations.
- Scikit-learn: Machine learning library used for training models.
- XGBoost: Optimized gradient boosting library.
- Joblib: Library for saving and loading models.
- HTML/CSS: For front-end development.

## Installation Instructions

To set up this project locally, follow these steps:

1. Clone the repository:
   ```
    git clone https://github.com/Nikesh264/AirQualityPrediction.git
   ```
   ```
    cd AirQualityPrediction
   ```

3. Create a virtual environment (optional but recommended):
 ```  
  python -m venv venv
```

5. Activate the virtual environment:
- For Windows:
  ```
  venv\Scripts\activate
  ```
- For macOS/Linux:
  ```
  source venv/bin/activate
  ```

4. Install the required dependencies:
   
```
pip install -r requirements.txt
```


6. Train the machine learning models:
   
```
 python train_models.py
```

## Usage Instructions

1. Run the Flask application:

```
python app.py
```

2. Open your web browser and navigate to `http://127.0.0.1:5000/`.

3. Input pollutant levels in the provided form and select the city.

4. Click on "Predict AQI" to view the predicted AQI, category, recommendations, and pollutant impact analysis.

