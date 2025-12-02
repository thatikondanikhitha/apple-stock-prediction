import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from datetime import timedelta

# -------------------------------
# Load Model and Scaler
# -------------------------------
MODEL_PATH = "xgb_model.pkl"       # change your filename
SCALER_PATH = "model_name.pkl"           # if used

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
    # df['Date'] = pd.to_datetime(df['Date'])
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', infer_datetime_format=True)
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

    import streamlit as st
import pandas as pd
import numpy as np
import pickle  # or use joblib if your model was saved with joblib
from datetime import timedelta

st.title("🍎 Apple Stock Prediction")

# -------------------------------
# Load the trained model once
# -------------------------------
@st.cache_resource
def load_model():
    with open("apple_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()
st.write("✅ Model loaded:", type(model))

# -------------------------------
# Load scaler if exists
# -------------------------------
try:
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    st.write("✅ Scaler loaded:", type(scaler))
except:
    scaler = None
    st.write("⚠ No scaler found. Using raw values.")

# -------------------------------
# CSV Upload
# -------------------------------
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded data preview:")
    st.dataframe(df.head())

    # Assume last row contains features for prediction
    last_value = df.iloc[-1].values  # numpy array
    current_value = last_value.reshape(1, -1)  # ensure correct 2D shape

    st.subheader("📊 Next 30 Days Prediction")
    predictions_scaled = []

    for _ in range(30):
        # Predict next day
        next_pred = model.predict(current_value)  # shape (1,) or (1, n_features)
        predictions_scaled.append(next_pred[0])

        # Update current_value for next prediction
        # If single feature, just reshape next_pred
        current_value = np.array(next_pred).reshape(1, -1)

    # Inverse scale if scaler exists
    if scaler is not None:
        predictions = scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1))
    else:
        predictions = np.array(predictions_scaled).reshape(-1, 1)

    # Prepare future dates
    last_date = pd.to_datetime(df['Date'].iloc[-1])
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












