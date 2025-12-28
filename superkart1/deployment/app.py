import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the trained model
model_path = hf_hub_download(repo_id="kamskathir/superkart_model", filename="best_superkart_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI
st.title("SuperKart future Sales Prediction")
st.write("""
This application predicts the expected **Sales** of SuperKart based on 
store location type, store type, store size, product sugar content and product type.
Please enter the  details below to get a sales prediction.
""")

# User input
product_type = st.selectbox("Product Type", ["Baking Goods", "Breads", "Breakfast", "Canned", "Diary", "Frozen Foods", "Fruits and Vegitables", "Hard Drinks", "Health and Hygiene","Household", "Meat", "Others","Seafood","Snack Foods", "Soft Drinks"])
product_sugar_content = st.selectbox("Low Sugar", "No Sugar", "Regular")
store_type = st.selectbox("Departmental Store", "Food Mart", "SuperMarket Type1", "SuperMarket Type2")
store_size = st.selectbox(["Low", "Medium", "High"])
store_location_city_type = st.selectbox("Tier 1", "Tier 2", "Tier 3")

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'product_type': product_type,
    'product_sugar_content': product_sugar_content,
    'store_type': store_type,
    'store_size': store_size,
    'store_location_city_type': store_location_city_type,
   }])

# Predict button
if st.button("Predict Sales Revenue"):
    prediction = model.predict(input_data)[0]
    st.subheader("Prediction Result:")
    st.success(f"Estimated Sales Revenue: **${prediction:,.2f} USD**")
