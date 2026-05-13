import streamlit as st
import pandas as pd
import pickle

# Load model
model = pickle.load(open('../notebooks/sales_model.pkl', 'rb'))

# Load columns
model_columns = pickle.load(open('../notebooks/model_columns.pkl', 'rb'))

st.title("Sales Prediction Dashboard")

st.write("Enter details below")

# Inputs
store_nbr = st.number_input("Store Number", min_value=1)

family = st.text_input("Product Family")

promotion = st.number_input("Promotion", min_value=0)

month = st.number_input("Month", min_value=1, max_value=12)

day = st.number_input("Day", min_value=1, max_value=31)

# Predict button
if st.button("Predict Sales"):

    input_data = pd.DataFrame({
        'store_nbr': [store_nbr],
        'family': [family],
        'onpromotion': [promotion],
        'month': [month],
        'day': [day]
    })

    # Convert text column
    input_data = pd.get_dummies(input_data)

    # Match training columns
    input_data = input_data.reindex(
        columns=model_columns,
        fill_value=0
    )

    # Predict
    prediction = model.predict(input_data)

    st.success(f"Predicted Sales: {prediction[0]:.2f}")