
# Accident Data Analysis and Prediction

This project provides a web-based tool for predicting and classifying road accidents based on various input parameters. It utilizes machine learning models to analyze accident severity given specific conditions.

## Technologies Used
- HTML
- CSS
- JavaScript
- Bootstrap
- Machine Learning:
  - CatBoostClassifier (from `catboost` library)
  - pandas
  - scikit-learn
- Data Visualization:
  - Matplotlib
  - Tabula

## Team Members
- Shivam Sharma
- Krishnan Lakshmi Narayana
- Bhavya Vishal
- Madhuri

## Usage Instructions

### Dashboard
1. Clone the repository:
   ```bash
   git clone https://github.com/theshivam7/Accident_data_analysis.git
   ```
2. Open `index.html` in a web browser to access the accident prediction dashboard.
3. Input relevant parameters such as police attendance, location coordinates, driver age, vehicle type, weather conditions, etc.
4. Click the "Predict" button to see the predicted accident severity.

### Machine Learning Model
1. Ensure you have Python installed on your machine.
2. Install required libraries:
   ```bash
   pip install pandas catboost scikit-learn matplotlib
   ```
3. Use the provided machine learning script (`ml_model.py`) to train and test the CatBoostClassifier model.
4. Example usage:
   ```python
   import pandas as pd
   from catboost import CatBoostClassifier
   from sklearn.model_selection import train_test_split
   import matplotlib.pyplot as plt
   from sklearn.metrics import accuracy_score
   import time

  
