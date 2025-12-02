import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from datetime import timedelta
import xgboost as xgb

st.set_page_config(page_title="Apple Stock Prediction", layout="wide")

@st.cache_resource
def load_model():
    try:
        with open("xgb_model.pkl", "rb") as f:
            model = pickle.load(f)
        st.success("✅ Model loaded!")
        st.info(f"Model expects: {getattr(model, 'n_features_in_', 'Unknown')} features")
        return model, None
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None, None

model, scaler = load_model()
if model is None:
    st.stop()

st.title("🍎 Apple Stock Prediction")
uploaded_file = st.file_uploader("📁 Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date']).sort_values("Date")
    
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(df.tail())
    with col2:
        st.metric("Last Close", f"${df['Close'].iloc[-1]:.2f}")
    
    # ✅ AUTO-PAD 6 FEATURES TO 10
    features = [col for col in df.columns if col != 'Date']  # ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    st.success(f"📊 Using {len(features)} features: {features}")
    
    # Pad to 10 features expected by model
    data = df[features].tail(1).values  # (1, 6)
    padded_data = np.pad(data, ((0,0),(0,4)), mode='constant')  # (1, 10) with 4 zeros
    
    last_value = padded_data[0]  # (10,)
    st.success(f"✅ Padded to {last_value.shape} ✓")
    
    # ✅ PREDICTION BUTTON
    if st.button("🚀 Predict Next 30 Days", type="primary"):
        predictions = []
        current_value = last_value.copy()
        
        st.info("🔄 Generating predictions...")
        progress_bar = st.progress(0)
        
        for i in range(30):
            try:
                pred = model.predict(current_value.reshape(1, -1))[0]
                predictions.append(pred)
                
                # Update for next step
                current_value[:6] = pred  # Update first 6 features
                current_value[6:] = 0     # Keep padding zeros
                
                progress_bar.progress(i+1)
                
            except Exception as e:
                st.error(f"❌ Step {i}: {str(e)}")
                predictions.append(np.nan)
        
        # ✅ DISPLAY RESULTS
        last_date = df['Date'].iloc[-1]
        future_dates = [last_date + timedelta(days=i+1) for i in range(30)]
        
        pred_df = pd.DataFrame({
            "Date": future_dates,
            "Predicted": predictions
        })
        
        st.subheader("📈 30-Day Forecast")
        st.dataframe(pred_df)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df["Date"], df["Close"], label="Historical Close", linewidth=2)
        ax.plot(pred_df["Date"], pred_df["Predicted"], 
                label="Forecast", linestyle="--", color="orange", linewidth=2)
        ax.axvline(x=last_date, color="red", linestyle=":", label="Forecast Start")
        ax.set_title("Apple Stock Price Forecast")
        ax.legend(); ax.grid(alpha=0.3)
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        # Stats
        valid_preds = pred_df["Predicted"].dropna()
        col1, col2, col3 = st.columns(3)
        col1.metric("Min", f"${valid_preds.min():.2f}")
        col2.metric("Avg", f"${valid_preds.mean():.2f}")
        col3.metric("Max", f"${valid_preds.max():.2f}")

else:
    st.info("👆 Upload your CSV to start predicting!")
