import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
import matplotlib.pyplot as plt
from datetime import timedelta

# -------------------------------
# Load Model and Scaler
# -------------------------------
MODEL_PATH = "xgb_model.pkl"       # change your filename
SCALER_PATH = "scaler.pkl"           # if used

@st.cache_resource
def load_model():
    try:
        model = pickle.load(open(MODEL_PATH, "rb"))
        scaler = pickle.load(open(SCALER_PATH, "rb"))
        return model, scaler
    except:
        return None, None

model, scaler = load_model()

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("📈 Apple Stock Price Prediction")
st.write("Predicting the next **30 days** stock prices using trained model.")

uploaded_file = st.file_uploader("Upload Stock Market CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    #df['Date'] = pd.to_datetime(df['Date'])
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce',dayfirst=False)
    df = df.dropna(subset=['Date'])

    df = df.sort_values("Date")

    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(df.tail())

    # -------------------------------------
    # Feature Preparation
    # -------------------------------------
    st.subheader("🔧 Feature Engineering")

    # Use only closing price (modify if you used more features)
    data = df[['Close']].values

    if scaler is not None:
        data_scaled = scaler.transform(data)
    else:
        data_scaled = data

    last_value = data_scaled[-1]

    # -------------------------------------
    # Predict Next 30 Days
    # -------------------------------------
    st.subheader("📊 Next 30 Days Prediction")

    predictions_scaled = []
    current_value = last_value

    for _ in range(30):
        next_pred = model.predict(current_value.reshape(1, -1))
        predictions_scaled.append(next_pred[0])
        current_value = next_pred

    # Inverse transform if scaler exists
    if scaler is not None:
        predictions = scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1))
    else:
        predictions = np.array(predictions_scaled).reshape(-1, 1)

    # Prepare dates
    last_date = df['Date'].iloc[-1]
    future_dates = [last_date + timedelta(days=i+1) for i in range(30)]

    pred_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted Close": predictions.flatten()
    })

    st.write(pred_df)

    # -------------------------------------
    # Graph
    # -------------------------------------
    st.subheader("📈 Prediction Visualization")

    plt.figure(figsize=(10, 5))
    plt.plot(df["Date"], df["Close"], label="Historical Close Price")
    plt.plot(pred_df["Date"], pred_df["Predicted Close"], label="Predicted Close", linestyle='--')
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    st.pyplot(plt)

else:
    st.info("👉 Please upload your Stock Market CSV file to continue.")



