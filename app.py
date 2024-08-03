# app.py

import streamlit as st
import numpy as np
import joblib

# Load the model and scaler
model = joblib.load('linear_regression.pkl')
feature_scaler = joblib.load('feature.pkl')

def predict_price(bedrooms, area, bathrooms):
    # Prepare the input data
    input_data = np.array([[area, bedrooms, bathrooms]])
    input_data_scaled = feature_scaler.transform(input_data)
    
    # Predict the price
    prediction = model.predict(input_data_scaled)
    return prediction[0]

# Streamlit app
st.title(' 🏠 House Price Prediction')

st.write('Enter the details of the house to predict the price.')

# User inputs
area = st.number_input('Area (sqft)', min_value=0, value=1500)
bedrooms = st.number_input('Number of Bedrooms', min_value=0, value=1)
bathrooms= st.number_input('Number of Bathrooms', min_value=0, value=1)
if st.button('Predict Price'):
    predicted_price = predict_price(area, bedrooms, bathrooms)
    st.write(f'Predicted House Price: ${predicted_price:,.2f}')
