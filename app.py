from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

app = Flask(__name__)

column_names = ['Area', 'Production', 'GDP', 'Annual Growth Rate', 'Inflation', 'Rainfall', 'Temperature']

data = pd.read_csv('./updated_dataset.csv')
X_train = data[column_names]
y_train = data['Target']

my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

X_train_scaled = my_pipeline.fit_transform(X_train)

model = RandomForestRegressor()
model.fit(X_train_scaled, y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    input_df = pd.DataFrame([input_features], columns=column_names)
    input_prepared = my_pipeline.transform(input_df)
    prediction = model.predict(input_prepared)
    output = round(prediction[0], 2)
    return render_template('index.html', prediction_text=f'Predicted Crop Price: ₹{output}')

if __name__ == "__main__":
    app.run(debug=True)
