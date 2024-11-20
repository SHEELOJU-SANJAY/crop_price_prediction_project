from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

app = Flask(__name__)

# Define column names for the input features
column_names = ['Area', 'Production', 'GDP', 'Annual Growth Rate', 'Inflation', 'Rainfall', 'Temperature']

# Load dataset from CSV
dataset = pd.read_csv('Dataset.csv')  # Replace with the actual path to your dataset

# Split the dataset into features (X) and target (y)
X_train = dataset[column_names].values  # Features
y_train = dataset['Crop Price'].values  # Target variable

# Initialize and fit the pipeline with training data
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),  # Handle missing values
    ('scaler', StandardScaler())                 # Scale the features
])

# Preprocess the training data
X_train_scaled = my_pipeline.fit_transform(X_train)

# Train the Random Forest model
model = RandomForestRegressor()
model.fit(X_train_scaled, y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract data from form as a list of float values
        input_features = [float(x) for x in request.form.values()]
        
        # Convert input features to DataFrame to ensure correct column names
        input_df = pd.DataFrame([input_features], columns=column_names)
        
        # Preprocess the input data with the fitted pipeline
        input_prepared = my_pipeline.transform(input_df)
        
        # Make the prediction
        prediction = model.predict(input_prepared)
        output = round(prediction[0], 2)
        return render_template('index.html', prediction_text=f'Predicted Crop Price: ₹{output}')
    except Exception as e:
        return render_template('index.html', prediction_text="Error: Invalid input data.")

if __name__ == "__main__":
    app.run()  # Remove debug=True for production
