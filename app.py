import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from keras.models import load_model
import pickle
import streamlit as st
import tensorflow as tf

# Load model and preprocessing objects
model = load_model('model.h5')
scaler = pickle.load(open('scaler.pkl', 'rb'))
ohe = pickle.load(open('ohe.pkl', 'rb'))
label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))

# App title and intro
st.set_page_config(page_title="Customer Churn Prediction", layout="centered")
st.markdown("<h1 style='text-align: center;'>Customer Churn Prediction</h1>", unsafe_allow_html=True)
st.markdown("Use this tool to predict whether a customer is likely to churn based on their banking profile.")

# --- Sidebar Inputs ---
st.sidebar.header("Customer Information")

geography = st.sidebar.selectbox('Geography', ohe.categories_[0])
gender = st.sidebar.selectbox('Gender', label_encoder.classes_)
age = st.sidebar.slider('Age', min_value=18, max_value=100, value=35)
tenure = st.sidebar.slider('Tenure (Years with Bank)', min_value=0, max_value=10, value=3)
balance = st.sidebar.number_input('Balance', min_value=0.0, max_value=100000.0, step=100.0, value=50000.0)
num_of_products = st.sidebar.slider('Number of Products', min_value=1, max_value=4, value=1)
credit_score = st.sidebar.number_input('Credit Score', min_value=300, max_value=850, step=1, value=650)
has_credit_card = st.sidebar.selectbox('Has Credit Card', ['Yes', 'No'])
is_active_member = st.sidebar.selectbox('Is Active Member', ['Yes', 'No'])
estimated_salary = st.sidebar.number_input('Estimated Salary', min_value=0.0, max_value=100000.0, step=100.0, value=60000.0)

# --- Data Preprocessing ---
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Geography': [geography],
    'Gender': [label_encoder.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [1 if has_credit_card == 'Yes' else 0],
    'IsActiveMember': [1 if is_active_member == 'Yes' else 0],
    'EstimatedSalary': [estimated_salary]
})

# Encode Geography
geo_encoded = ohe.transform([[input_data['Geography'][0]]])
geo_encoded_df = pd.DataFrame(geo_encoded, columns=ohe.get_feature_names_out(['Geography']))

# Final input frame
input_df = pd.concat([input_data.drop(columns=['Geography']), geo_encoded_df], axis=1)
input_df = input_df[scaler.feature_names_in_]  # match column order

# --- Prediction ---
input_data_scaled = scaler.transform(input_df)
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]
result_text = "The customer is likely to churn." if prediction_proba > 0.5 else "The customer is not likely to churn."

# --- Output Section ---
st.markdown("---")
st.subheader("Prediction Result")
st.write(result_text)
st.metric(label="Churn Probability", value=f"{prediction_proba:.2f}")
