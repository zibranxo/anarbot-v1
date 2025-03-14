import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import pickle

# Load preprocessed data
padded_sequences = np.load("padded_sequences.npy")
encoded_tags = np.load("encoded_tags.npy")

# Load tokenizer and label encoder
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
with open("label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# Model parameters
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 16
max_length = padded_sequences.shape[1]
output_size = len(set(encoded_tags))

# Build LSTM model
model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_length),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dense(32, activation="relu"),
    Dropout(0.2),
    Dense(output_size, activation="softmax")
])

# Compile model
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train model
model.fit(padded_sequences, encoded_tags, epochs=100, batch_size=8)

# Save model
model.save("chatbot_lstm.h5")
print("Model trained and saved successfully!")
