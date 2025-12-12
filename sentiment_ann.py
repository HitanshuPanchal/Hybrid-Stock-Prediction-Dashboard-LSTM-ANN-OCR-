import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Dropout, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
df = pd.read_csv("augmented_5000_paraphrased.csv")
df.dropna(inplace=True)

texts = df["text"].astype(str).values
labels = df["sentiment"].values

# ------------------------------------------------------------
# TOKENIZER
# ------------------------------------------------------------
vocab_size = 10000
max_len = 50

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)
padded = pad_sequences(sequences, maxlen=max_len, padding="post")

# SAVE TOKENIZER + CONFIG
pickle.dump(tokenizer, open("tokenizer.pkl", "wb"))
pickle.dump({"max_len": max_len, "vocab_size": vocab_size}, open("config.pkl", "wb"))

# ------------------------------------------------------------
# TRAIN-TEST SPLIT
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    padded, labels, test_size=0.2, random_state=42
)

# ------------------------------------------------------------
# ANN MODEL
# ------------------------------------------------------------
model = Sequential([
    Embedding(vocab_size, 64, input_length=max_len),
    Dropout(0.3),
    GlobalAveragePooling1D(),
    Dense(128, activation="relu"),
    Dropout(0.4),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# ------------------------------------------------------------
# TRAIN MODEL
# ------------------------------------------------------------
es = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=15,
    batch_size=32,
    callbacks=[es]
)

# ------------------------------------------------------------
# SAVE MODEL
# ------------------------------------------------------------
model.save("sentiment_model.h5")
print("\nModel saved as sentiment_model.h5")

# ------------------------------------------------------------
# EVALUATE
# ------------------------------------------------------------
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {acc * 100:.2f}%")
