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
st.markdown("Upload Apple stock CSV with **Date** and **Close** columns to predict next 30 days.")

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

    # Feature preparation
    data = df[['Close']].values
    if scaler is not None:
        data_scaled = scaler.transform(data)
    else:
        data_scaled = data
    
    last_value = data_scaled[-1, 0]  # Single value for univariate
    
    # -------------------------------
    # Prediction Loop (Fixed for XGBRegressor)
    # -------------------------------
    st.subheader("🔮 Next 30 Days Prediction")
    
    if st.button("🚀 Generate Predictions", type="primary"):
        predictions_scaled = []
        current_value = np.array([[last_value]])  # Shape (1,1)
        
        with st.spinner("Generating predictions..."):
            for i in range(30):
                try:
                    # Use numpy array directly for XGBRegressor
                    pred = model.predict(current_value)[0]
                    predictions_scaled.append(pred)
                    current_value = np.array([[pred]])  # Update for next step
                except Exception as e:
                    st.error(f"Prediction failed at step {i}: {str(e)}")
                    break
        
        # Inverse transform
        if scaler is not None:
            predictions = scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1)).flatten()
        else:
            predictions = np.array(predictions_scaled)
        
        # Create results
        last_date = df['Date'].iloc[-1]
        future_dates = [last_date + timedelta(days=i+1) for i in range(30)]
        
        pred_df = pd.DataFrame({
            "Date": future_dates,
            "Predicted Close": predictions
        })
        
        # Display results
        st.dataframe(pred_df)
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Min Prediction", f"${predictions.min():.2f}")
        with col2:
            st.metric("Avg Prediction", f"${predictions.mean():.2f}")
        with col3:
            st.metric("Max Prediction", f"${predictions.max():.2f}")
        
        # -------------------------------
        # Visualization
        # -------------------------------
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df["Date"], df["Close"], label="Historical Close", linewidth=2)
        ax.plot(pred_df["Date"], pred_df["Predicted Close"], 
                label="Predicted Close", linestyle='--', linewidth=2, color='orange')
        ax.axvline(x=last_date, color='red', linestyle=':', alpha=0.7, label="Prediction Start")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price ($)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

else:
    st.info("👆 Please upload your Apple stock CSV file with Date and Close columns.")
    st.markdown("**Sample CSV format:**")
    sample_data = pd.DataFrame({
        'Date': ['2025-01-01', '2025-01-02', '2025-01-03'],
        'Close': [150.25, 152.10, 149.80]
    })
    st.dataframe(sample_data)

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit + XGBoost")

