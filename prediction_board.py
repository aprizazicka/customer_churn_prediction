import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time

# App title
st.title("☎️ Customer Churn Prediction App")

# Sidebar for user input
st.sidebar.title("📝 Input Customer Data")
st.sidebar.write("Masukkan data Anda untuk masing-masing fitur di bawah ini:")

def input_data_user():
    # Encode state using a simple index (you can adjust based on your training setup)
    states = ['OH', 'NJ', 'OK', 'MA', 'MO', 'LA', 'WV', 'IN', 'RI', 'IA', 'MT', 'NY', 'ID',
              'VA', 'TX', 'FL', 'CO', 'AZ', 'SC', 'WY', 'HI', 'NH', 'AK', 'GA', 'MD', 'AR',
              'WI', 'OR', 'MI', 'DE', 'UT', 'CA', 'SD', 'NC', 'WA', 'MN', 'NM', 'NV', 'DC',
              'VT', 'KY', 'ME', 'MS', 'AL', 'NE', 'KS', 'TN', 'IL', 'PA', 'CT', 'ND']
    state = st.sidebar.selectbox("State", states)
    state_encoded = states.index(state)

    area_codes = {"area_code_408": 408, "area_code_415": 415, "area_code_510": 510}
    area_code_label = st.sidebar.selectbox("Area Code", list(area_codes.keys()))
    area_code = area_codes[area_code_label]

    account_length = st.sidebar.number_input("Account Length", 0, 300, value=100)

    international_plan = st.sidebar.radio("Subscribe International Plan", ["Yes", "No"])
    voice_mail_plan = st.sidebar.radio("Subscribe Voice Mail Plan", ["Yes", "No"])
    number_vmail_messages = st.sidebar.number_input("Total Voicemail Messages", 0, 60, value=10)

    total_day_minutes = st.sidebar.number_input("Total Day Minutes", 0.0, 400.0)
    total_day_calls = st.sidebar.number_input("Total Day Calls", 0, 200)
    total_day_charge = st.sidebar.number_input("Total Day Charge", 0.0, 70.0)

    total_eve_minutes = st.sidebar.number_input("Total Evening Minutes", 0.0, 400.0)
    total_eve_calls = st.sidebar.number_input("Total Evening Calls", 0, 200)
    total_eve_charge = st.sidebar.number_input("Total Evening Charge", 0.0, 70.0)

    total_night_minutes = st.sidebar.number_input("Total Night Minutes", 0.0, 400.0)
    total_night_calls = st.sidebar.number_input("Total Night Calls", 0, 200)
    total_night_charge = st.sidebar.number_input("Total Night Charge", 0.0, 70.0)

    total_intl_minutes = st.sidebar.number_input("Total International Minutes", 0.0, 30.0)
    total_intl_calls = st.sidebar.number_input("Total International Calls", 0, 20)
    total_intl_charge = st.sidebar.number_input("Total International Charge", 0.0, 10.0)

    number_customer_service_calls = st.sidebar.number_input("Customer Service Calls", 0, 10)

    # Convert Yes/No to 1/0
    voice_mail_plan_num = 1 if voice_mail_plan == "Yes" else 0
    international_plan_num = 1 if international_plan == "Yes" else 0

    # All numeric inputs, matching model training format
    features = np.array([[
        state_encoded, account_length, area_code, international_plan_num,
        voice_mail_plan_num, number_vmail_messages, total_day_minutes,
        total_day_calls, total_day_charge, total_eve_minutes, total_eve_calls,
        total_eve_charge, total_night_minutes, total_night_calls,
        total_night_charge, total_intl_minutes, total_intl_calls,
        total_intl_charge, number_customer_service_calls
    ]])

    return features

# Get numeric input vector
input_features = input_data_user()

# Predict
if st.button("📈 Prediksi", use_container_width=True):
    st.subheader("🔍 Hasil Prediksi")

    try:
        with open("xgb_model_smote.pkl", 'rb') as file:
            loaded_model = pickle.load(file)

        with st.spinner("⏳ Memprediksi..."):
            time.sleep(2)
            prediction = loaded_model.predict(input_features)

            if prediction[0] == 1:
                result_html = """
                <div style="background-color:#7f1d1d; color:white; padding:1rem; border-radius:10px;
                            font-size:18px; font-weight:bold;">
                    🔥 Diprediksi Berpotensi Melakukan Churn
                </div>
                """
            else:
                result_html = """
                <div style="background-color:#14532d; color:white; padding:1rem; border-radius:10px;
                            font-size:18px; font-weight:bold;">
                    ✅ Pelanggan Diprediksi Tetap Bertahan
                </div>
                """

            st.markdown(result_html, unsafe_allow_html=True)

    except FileNotFoundError:
        st.error("❌ Model tidak ditemukan. Pastikan file `xgb_model_smote.pkl` tersedia di direktori.")
    except Exception as e:
        st.error(f"❌ Terjadi error saat memuat model: {e}")