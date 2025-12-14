import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import pickle
import re
import unicodedata
import easyocr
import os

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
import plotly.graph_objs as go

# ============================================================
# RENDER DISK PATH (VERY IMPORTANT)
# ============================================================
MODEL_DIR = "/opt/render/project/data"

LSTM_MODEL_PATH = os.path.join(MODEL_DIR, "lstm_model.h5")
SENTIMENT_MODEL_PATH = os.path.join(MODEL_DIR, "sentiment_model.h5")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pkl")
CONFIG_PATH = os.path.join(MODEL_DIR, "config.pkl")

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Hybrid Stock Prediction Dashboard",
    layout="wide"
)

st.title("📈 Hybrid Stock Prediction Dashboard")
st.caption("LSTM Price Prediction • ANN Sentiment • OCR News Analysis")

# ============================================================
# TEXT CLEANING
# ============================================================
def clean_text(text):
    text = unicodedata.normalize("NFKD", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[^a-zA-Z0-9.,!? ]+", "", text)
    text = re.sub(r"\b[a-zA-Z]\b", "", text)
    return text

# ============================================================
# OCR INITIALIZATION
# ============================================================
@st.cache_resource
def load_ocr():
    return easyocr.Reader(["en"])

reader = load_ocr()

# ============================================================
# SIDEBAR INPUTS
# ============================================================
st.sidebar.header("📌 Stock Selection")

ticker = st.sidebar.text_input("Stock Ticker", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2025-01-01"))

# ============================================================
# SENTIMENT INPUT
# ============================================================
st.sidebar.header("📰 Sentiment Analysis")

if "news_text" not in st.session_state:
    st.session_state.news_text = ""

uploaded_image = st.sidebar.file_uploader(
    "Upload News Image (OCR)", type=["png", "jpg", "jpeg"]
)

if uploaded_image:
    extracted = reader.readtext(uploaded_image.read(), detail=0)
    cleaned = clean_text(" ".join(extracted))
    if cleaned:
        st.session_state.news_text = cleaned
        st.sidebar.success("OCR text extracted")

news_input = st.sidebar.text_area(
    "Enter News Text", key="news_text"
)

sentiment_ready = news_input.strip() != ""

# ============================================================
# LOAD STOCK DATA
# ============================================================
df = yf.download(ticker, start=start_date, end=end_date)

if df.empty:
    st.error("❌ No stock data found.")
    st.stop()

st.subheader("📊 Historical Stock Data")
st.dataframe(df.tail(50))

# ============================================================
# TECHNICAL INDICATORS
# ============================================================
df["MA20"] = df["Close"].rolling(20).mean()

delta = df["Close"].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()
df["RSI"] = 100 - (100 / (1 + avg_gain / avg_loss))

ema12 = df["Close"].ewm(span=12, adjust=False).mean()
ema26 = df["Close"].ewm(span=26, adjust=False).mean()
df["MACD"] = ema12 - ema26
df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

df.dropna(inplace=True)

# ============================================================
# LSTM FEATURE PREP
# ============================================================
features = df[
    ["Open", "High", "Low", "Close", "Volume",
     "MA20", "RSI", "MACD", "Signal"]
]

scaler = MinMaxScaler()
scaled = scaler.fit_transform(features.values)

SEQ_LEN = 60

X, y = [], []
for i in range(SEQ_LEN, len(scaled)):
    X.append(scaled[i-SEQ_LEN:i])
    y.append(scaled[i, 3])

X, y = np.array(X), np.array(y)

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ============================================================
# LOAD OR TRAIN LSTM
# ============================================================
st.sidebar.header("⚙️ Model Control")
train_lstm = st.sidebar.checkbox("Train LSTM (slow)", False)

if train_lstm:
    lstm_model = Sequential([
        LSTM(64, return_sequences=True,
             input_shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(64),
        Dense(1)
    ])

    lstm_model.compile(optimizer="adam", loss="mse")
    st.write("⏳ Training LSTM...")
    lstm_model.fit(X_train, y_train, epochs=10, batch_size=32)
    lstm_model.save(LSTM_MODEL_PATH)
    st.success("✅ LSTM model saved")

else:
    if not os.path.exists(LSTM_MODEL_PATH):
        st.error("❌ LSTM model not found on Render Disk.")
        st.stop()

    lstm_model = load_model(LSTM_MODEL_PATH)

# ============================================================
# LSTM PREDICTION
# ============================================================
pred_scaled = lstm_model.predict(X_test, verbose=0)

close_scaler = MinMaxScaler()
close_scaler.min_ = np.array([scaler.min_[3]])
close_scaler.scale_ = np.array([scaler.scale_[3]])

predictions = close_scaler.inverse_transform(pred_scaled)
actual = close_scaler.inverse_transform(y_test.reshape(-1, 1))

# ============================================================
# PRICE PLOT
# ============================================================
st.subheader("📈 Actual vs Predicted Price")

fig = go.Figure()
fig.add_trace(go.Scatter(
    y=actual.flatten(),
    mode="lines",
    name="Actual"
))
fig.add_trace(go.Scatter(
    y=predictions.flatten(),
    mode="lines",
    name="Predicted"
))
fig.update_layout(height=500)
st.plotly_chart(fig, use_container_width=True)

# ============================================================
# LOAD SENTIMENT MODEL
# ============================================================
if not all(map(os.path.exists,
               [SENTIMENT_MODEL_PATH, TOKENIZER_PATH, CONFIG_PATH])):
    st.error("❌ Sentiment model files missing on disk.")
    st.stop()

sentiment_model = load_model(SENTIMENT_MODEL_PATH)
tokenizer = pickle.load(open(TOKENIZER_PATH, "rb"))
config = pickle.load(open(CONFIG_PATH, "rb"))
max_len = config["max_len"]

# ============================================================
# SENTIMENT ANALYSIS
# ============================================================
st.subheader("🧠 Sentiment Analysis Result")

if sentiment_ready:
    seq = tokenizer.texts_to_sequences([news_input])
    padded = pad_sequences(seq, maxlen=max_len, padding="post")

    prob = sentiment_model.predict(padded, verbose=0)[0][0]
    score = prob * 100

    if prob > 0.6:
        label, color = "Positive 😊", "green"
    elif prob < 0.4:
        label, color = "Negative 😟", "red"
    else:
        label, color = "Neutral 😐", "orange"

    st.markdown(
        f"<h3 style='color:{color}'>{label}</h3>",
        unsafe_allow_html=True
    )
    st.write(f"Sentiment Score: {score:.2f}%")

# ============================================================
# HYBRID SIGNAL
# ============================================================
st.subheader("📌 Hybrid Trading Signal")

last_actual = actual[-1][0]
last_pred = predictions[-1][0]

price_change = (last_pred - last_actual) / last_actual * 100
combined_score = 0.7 * price_change + 0.3 * (score / 100)

if combined_score > 0.2:
    signal, sig_color = "BUY", "green"
elif combined_score < -0.2:
    signal, sig_color = "SELL", "red"
else:
    signal, sig_color = "HOLD", "orange"

st.markdown(
    f"<h2 style='color:{sig_color}; text-align:center'>{signal}</h2>",
    unsafe_allow_html=True
)

# ============================================================
# 7-DAY FORECAST
# ============================================================
st.subheader("🔮 7-Day Forecast")

future_prices = []
last_seq = scaled[-SEQ_LEN:].copy()

for _ in range(7):
    pred = lstm_model.predict(
        last_seq.reshape(1, SEQ_LEN, features.shape[1]),
        verbose=0
    )
    price = close_scaler.inverse_transform(pred)[0][0]
    future_prices.append(price)

    new_row = last_seq[-1].copy()
    new_row[3] = pred[0][0]
    last_seq = np.vstack([last_seq[1:], new_row])

forecast_df = pd.DataFrame({
    "Day": range(1, 8),
    "Forecast Price": future_prices
})

st.bar_chart(forecast_df.set_index("Day"))

# ============================================================
# CONFIDENCE SCORE
# ============================================================
errors = np.abs(predictions - actual) / actual
confidence = max(0, min((1 - np.mean(errors)) * 100, 100))

st.subheader("🎯 Model Confidence")
st.progress(int(confidence))
st.write(f"Confidence: {confidence:.2f}%")
