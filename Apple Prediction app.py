import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from datetime import timedelta
import xgboost as xgb

# Page config
st.set_page_config(page_title="Apple Stock Prediction", layout="wide")

# -------------------------------
# Load Model and Scaler
# -------------------------------
@st.cache_resource
def load_model():
    MODEL_PATH = "xgb_model.pkl"
    SCALER_PATH = "scaler.pkl"
    
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        st.success("✅ Model loaded!")
        
        scaler = None
        try:
            with open(SCALER_PATH, "rb") as f:
                scaler = pickle.load(f)
            st.success("✅ Scaler loaded!")
        except FileNotFoundError:
            st.warning("⚠️ No scaler. Using raw data.")
        
        st.info(f"Model expects: {model.n_features_in_} features")
        return model, scaler
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.stop()

model, scaler = load_model()

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🍎 Apple Stock Prediction")
uploaded_file = st.file_uploader("📁 Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date']).sort_values("Date")
    
    st.subheader("📊 Data Preview")
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(df.tail())
    with col2:
        if 'Close' in df.columns:
            st.metric("Last Close", f"${df['Close'].iloc[-1]:.2f}")
    
    # ✅ AUTO-DETECT FEATURES FROM YOUR CSV (5 columns)
    available_features = [col for col in df.columns if col != 'Date']
    st.info(f"📈 Detected {len(available_features)} features: {available_features}")
    
    # Check feature count mismatch
    expected_features = model.n_features_in_
    if len(available_features) != expected_features:
        st.error(f"❌ MISMATCH: Model expects {expected_features} features, CSV has {len(available_features)}")
        st.info("💡 Solution: Retrain model with your 5 CSV features OR pad missing features with zeros")
        st.stop()
    
    # ✅ Use ALL available features from CSV
    data = df[available_features].tail(1).values  # Shape: (1, 5)
    
    if scaler:
        data_scaled = scaler.transform(data)
    else:
        data_scaled = data
    
    last_value = data_scaled[0]  # Shape: (5,)
    st.success(f"✅ Input ready: shape {last_value.shape}")
    
    # Prediction
    if st.button("🚀 Predict 30 Days", type="primary"):
        predictions = []
        current_value = last_value.copy()
        
        with st.spinner("Predicting..."):
            for i in range(30):
                try:
                    pred = model.predict(current_value.reshape(1, -1))[0]
                    predictions.append(pred)
                    successful_predictions += 1
                    # Simple autoregressive: use prediction for all features
                    current_value = np.zero(len(excepted_features)
                    current_value[:len(csv_features)] = pred 
                except Exception as e:
                    st.error(f"Step {i}: {str(e)}")
                    break
        
        # Results
        last_date = df['Date'].iloc[-1]
        future_dates = [last_date + timedelta(days=i+1) for i in range(30)]
        
        pred_df = pd.DataFrame({
            "Date": future_dates,
            "Predicted": predictions[:len(future_dates)]  # ✅ Fix length mismatch
        })
        
        st.subheader("📈 Predictions")
        st.dataframe(pred_df)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        if 'Close' in df.columns:
            ax.plot(df["Date"], df["Close"], label="Historical", linewidth=2)
        ax.plot(pred_df["Date"], pred_df["Predicted"], 
                label="Predicted", linestyle="--", color="orange", linewidth=2)
        ax.set_xlabel("Date"); ax.set_ylabel("Price"); ax.legend(); ax.grid(alpha=0.3)
        plt.xticks(rotation=45); st.pyplot(fig)

else:
    st.info("👆 Upload CSV with Date + your 5 features")

