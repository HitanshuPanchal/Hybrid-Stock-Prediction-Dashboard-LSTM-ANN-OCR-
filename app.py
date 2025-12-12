import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import pickle
import re
import unicodedata
import easyocr
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
import plotly.graph_objs as go
import os

# -----------------------
# CONFIG
# -----------------------
MODEL_DIR = os.getenv("MODEL_DIR", "/opt/render/project/data")
# Also allow local development fallback
LOCAL_MODEL_DIR = "./models"

# helper to resolve model path (persistent mount preferred)
def model_path(filename):
    path1 = os.path.join(MODEL_DIR, filename)
    path2 = os.path.join(LOCAL_MODEL_DIR, filename)
    if os.path.exists(path1):
        return path1
    if os.path.exists(path2):
        return path2
    # fallback to cwd
    return filename

# -----------------------
# STREAMLIT PAGE CONFIG
# -----------------------
st.set_page_config(page_title="Hybrid Stock Prediction Dashboard", layout="wide")
st.title("Hybrid Stock Prediction Dashboard (LSTM + ANN + OCR)")

# -----------------------
# UTILITIES
# -----------------------
def clean_text(text):
    text = unicodedata.normalize("NFKD", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[^a-zA-Z0-9.,!? ]+", "", text)
    text = re.sub(r"\b[a-zA-Z]\b", "", text)
    text = re.sub(' +', ' ', text)
    return text

# Use caching for heavy objects
@st.cache_resource
def get_easyocr_reader():
    try:
        reader = easyocr.Reader(["en"], gpu=False)
        return reader
    except Exception as e:
        st.error(f"EasyOCR init failed: {e}")
        return None

@st.cache_resource
def load_keras_model_safe(path):
    return load_model(path)

# -----------------------
# SIDEBAR
# -----------------------
st.sidebar.header("Stock & Date Selection")
ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2025-01-01"))

st.sidebar.header("Sentiment Analysis (ANN)")
if "news_text" not in st.session_state:
    st.session_state["news_text"] = ""

uploaded_image = st.sidebar.file_uploader("Upload News Image", type=["png", "jpg", "jpeg"])
if uploaded_image:
    st.sidebar.image(uploaded_image, width=250)
    reader = get_easyocr_reader()
    if reader is None:
        st.sidebar.error("OCR engine not available")
    else:
        extracted = reader.readtext(uploaded_image.read(), detail=0)
        cleaned = clean_text(" ".join(extracted))
        if cleaned.strip():
            st.session_state["news_text"] = cleaned
            st.sidebar.success("OCR text loaded!")
        else:
            st.sidebar.warning("No readable text found.")

news_input = st.sidebar.text_area("Enter News Text:", key="news_text")
sentiment_ready = news_input.strip() != ""

# -----------------------
# LOAD STOCK DATA
# -----------------------
with st.spinner("Downloading stock data..."):
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)

if df.empty:
    st.error("No stock data found.")
    st.stop()

st.subheader("📊 Historical Stock Data")
st.dataframe(df.tail(50))

# -----------------------
# ENGINEERING
# -----------------------
# technical indicators
try:
    df["MA20"] = df["Close"].rolling(20).mean()
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + (avg_gain / avg_loss)))
    EMA12 = df["Close"].ewm(span=12, adjust=False).mean()
    EMA26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = EMA12 - EMA26
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df.dropna(inplace=True)
except Exception as e:
    st.error(f"Indicator calc failed: {e}")
    st.stop()

# -----------------------
# LSTM FEATURES
# -----------------------
features = df[["Open", "High", "Low", "Close", "Volume", "MA20", "RSI", "MACD", "Signal"]]

data = features.values
scaler = MinMaxScaler()
scaled = scaler.fit_transform(data)

seq_len = 60
X, y = [], []
for i in range(seq_len, len(scaled)):
    X.append(scaled[i-seq_len:i])
    y.append(scaled[i, 3])
X, y = np.array(X), np.array(y)
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# -----------------------
# LSTM MODEL (load only)
# -----------------------
train_lstm = st.sidebar.checkbox("Train LSTM Model (slow)", False)
if train_lstm:
    lstm_model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(64),
        Dense(1)
    ])
    lstm_model.compile(optimizer='adam', loss='mse')
    st.write("Training LSTM model...")
    lstm_model.fit(X_train, y_train, epochs=15, batch_size=32)
    save_path = model_path('lstm_model.h5')
    lstm_model.save(save_path)
    st.success(f"LSTM model saved to {save_path}")
else:
    try:
        lstm_model = load_keras_model_safe(model_path('lstm_model.h5'))
    except Exception as e:
        st.error(f"No LSTM model found: {e}")
        st.stop()

# -----------------------
# PREDICTION & PLOTTING
# -----------------------
pred_scaled = lstm_model.predict(X_test)
close_scaler = MinMaxScaler()
close_scaler.min_, close_scaler.scale_ = scaler.min_[3], scaler.scale_[3]
predictions = close_scaler.inverse_transform(pred_scaled)
actual = close_scaler.inverse_transform(y_test.reshape(-1, 1))

st.subheader("📈 Actual vs Predicted")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index[seq_len+split:], y=actual.flatten(), mode='lines', name='Actual'))
fig.add_trace(go.Scatter(x=df.index[seq_len+split:], y=predictions.flatten(), mode='lines', name='Predicted'))
fig.update_layout(height=500)
st.plotly_chart(fig, use_container_width=True)

# -----------------------
# LOAD SENTIMENT MODEL
# -----------------------
try:
    sentiment_model = load_keras_model_safe(model_path('sentiment_model.h5'))
    with open(model_path('tokenizer.pkl'), 'rb') as f:
        tokenizer = pickle.load(f)
    with open(model_path('config.pkl'), 'rb') as f:
        config = pickle.load(f)
    max_len = config.get('max_len', 50)
except Exception as e:
    st.error(f"Sentiment model files missing or failed to load: {e}")
    st.stop()

# -----------------------
# SENTIMENT ANALYSIS
# -----------------------
st.subheader("📰 Sentiment Analysis Result")
if sentiment_ready:
    seq = tokenizer.texts_to_sequences([news_input])
    padded = pad_sequences(seq, maxlen=max_len, padding='post')
    prob = float(sentiment_model.predict(padded)[0][0])
    score = prob * 100
    if prob > 0.6:
        label, color = "Positive 😊", "green"
    elif prob < 0.4:
        label, color = "Negative 😟", "red"
    else:
        label, color = "Neutral 😐", "orange"
    st.markdown(f"<h3 style='color:{color}'>{label}</h3>", unsafe_allow_html=True)
    st.write(f"Sentiment Score: {score:.2f}%")
else:
    st.warning("Enter text or upload an image for sentiment analysis.")

# -----------------------
# HYBRID SIGNAL
# -----------------------
st.subheader("📌 Hybrid Trading Signal")
last_actual = actual[-1][0]
last_pred = predictions[-1][0]
price_change = (last_pred - last_actual) / last_actual * 100
combined_score = (0.7 * price_change) + (0.3 * (score / 100))
if combined_score > 0.2:
    signal, sig_color = "BUY", "green"
elif combined_score < -0.2:
    signal, sig_color = "SELL", "red"
else:
    signal, sig_color = "HOLD", "orange"
st.markdown(f"<h2 style='color:{sig_color}; text-align:center'>{signal}</h2>", unsafe_allow_html=True)

# -----------------------
# CONFIDENCE
# -----------------------
errors = np.abs(predictions - actual) / actual
confidence = max(0, min((1 - np.mean(errors)) * 100, 100))
st.subheader("🎯 Model Confidence Score")
st.progress(int(confidence))
st.write(f"Confidence: {confidence:.2f}%")

# -----------------------
# FUTURE FORECAST (optimized)
# -----------------------
st.subheader("🔮 7-Day Forecast")
future = []
last_seq = scaled[-seq_len:].copy()
for _ in range(7):
    pred_scaled = lstm_model.predict(last_seq.reshape(1, seq_len, features.shape[1]), verbose=0)
    pred_price = close_scaler.inverse_transform(pred_scaled)[0][0]
    future.append(pred_price)
    new_row = last_seq[-1].copy()
    new_row[3] = pred_scaled[0][0]
    last_seq = np.vstack([last_seq[1:], new_row])

# Build dates for forecast
last_date = df.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=7, freq='D')

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=df.index, y=df["Close"], mode='lines', name='Historical'))
fig2.add_trace(go.Scatter(x=future_dates, y=future, mode='lines+markers', name='7-Day Forecast', line=dict(dash='dash')))
fig2.update_layout(title='Stock Price History + 7-Day Forecast', xaxis_title='Date', yaxis_title='Price', template='plotly_white', height=500)
st.plotly_chart(fig2, use_container_width=True)

vol = np.std(future)
mean = np.mean(future)
forecast_conf = max(0, min((1 - (vol / mean)) * 100, 100))
st.subheader("🔎 Forecast Confidence")
st.progress(int(forecast_conf))
st.write(f"Forecast Confidence: {forecast_conf:.2f}%")
