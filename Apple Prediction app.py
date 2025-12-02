import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from datetime import timedelta
import xgboost as xgb

# Page config
st.set_page_config(page_title="Apple Stock Price Prediction", layout="wide")

# -------------------------------
# Load Model and Scaler with Debug
# -------------------------------
@st.cache_resource
def load_model():
    MODEL_PATH = "xgb_model.pkl"
    SCALER_PATH = "scaler.pkl"
    
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        st.success("✅ Model loaded successfully!")
        
        scaler = None
        try:
            with open(SCALER_PATH, "rb") as f:
                scaler = pickle.load(f)
            st.success("✅ Scaler loaded successfully!")
        except FileNotFoundError:
            st.warning("⚠️ Scaler not found. Using raw predictions.")
        
        # Debug info
        st.info(f"Model type: {type(model)}")
        return model, scaler
        
    except FileNotFoundError as e:
        st.error(f"❌ Missing file: {e}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Load error: {str(e)}")
        st.stop()

model, scaler = load_model()

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🍎 Apple Stock Price Prediction")
st.markdown("Upload Apple stock CSV with required features to predict next 30 days.")

# File uploader
uploaded_file = st.file_uploader("📁 Upload Stock CSV", type=["csv"])

if uploaded_file is not None:
    # Load and process data
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date']).sort_values("Date")
    
    st.subheader("📊 Data Preview")
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(df.tail(5))
    with col2:
        st.metric("Last Close Price", f"${df['Close'].iloc[-1]:.2f}")

    # -------------------------------
    # Feature Preparation - MULTIFeature Fix
    # -------------------------------
    st.subheader("🔧 Feature Engineering")

    FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume', 'Feature6', 'Feature7', 'Feature8', 'Feature9', 'Feature10']  # Replace with your actual 10 features

    missing_features = [f for f in FEATURES if f not in df.columns]
    if missing_features:
        st.error(f"❌ Missing features in CSV: {missing_features}")
        st.stop()

    data = df[FEATURES].tail(1).values  # shape (1, 10)
    
    if scaler is not None:
        data_scaled = scaler.transform(data)
    else:
        data_scaled = data

    last_value = data_scaled[0]  # shape (10,)
    st.write(f"Input feature shape: {last_value.shape}")

    # -------------------------------
    # Prediction Loop
    # -------------------------------
    st.subheader("🔮 Next 30 Days Prediction")
    
    if st.button("🚀 Generate Predictions", type="primary"):
        predictions_scaled = []
        current_value = last_value
        
        with st.spinner("Generating predictions..."):
            for i in range(30):
                try:
                    pred = model.predict(current_value.reshape(1, -1))[0]
                    predictions_scaled.append(pred)
                    
                    # Update current_value properly,
                    # here simplistic update - repeat predicted value for all features
                    current_value = np.full_like(current_value, pred)
                    
                except Exception as e:
                    st.error(f"Prediction failed at step {i}: {str(e)}")
                    break



