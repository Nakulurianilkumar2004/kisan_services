import numpy as np
import pandas as pd
import streamlit as st
import joblib

# Load the model and scaler
rfc = joblib.load('crop_recommendation_model.h5')
ms = joblib.load('scaler.h5')

# Crop dictionary
crop_dict = {
    1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut",
    6: "Papaya", 7: "Orange", 8: "Apple", 9: "Muskmelon", 10: "Watermelon",
    11: "Grapes", 12: "Mango", 13: "Banana", 14: "Pomegranate", 15: "Lentil",
    16: "Blackgram", 17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas",
    20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
}

# Streamlit UI
st.title("Crop Recommendation System")

# User input form
with st.form(key='crop_form'):
    N = st.number_input("Nitrogen (N)", min_value=0)
    P = st.number_input("Phosphorus (P)", min_value=0)
    k = st.number_input("Potassium (K)", min_value=0)
    temperature = st.number_input("Temperature (°C)", min_value=0.0)
    humidity = st.number_input("Humidity (%)", min_value=0)
    ph = st.number_input("pH", min_value=0.0)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0)
    
    submit_button = st.form_submit_button("Recommend Crop")

# Recommendation function
def recommend_crop(N, P, k, temperature, humidity, ph, rainfall):
    features = np.array([[N, P, k, temperature, humidity, ph, rainfall]])
    transformed_features = ms.transform(features)
    prediction = rfc.predict(transformed_features)
    return prediction[0]

if submit_button:
    predict = recommend_crop(N, P, k, temperature, humidity, ph, rainfall)
    
    if predict in crop_dict:
        crop_recommendation = crop_dict[predict]
        st.success(f"The best crop to cultivate is: {crop_recommendation}")
    else:
        st.error("Sorry, we are not able to recommend a proper crop for this environment.")

# To run the app, use the command:
# streamlit run crop_recommendation_app.py

