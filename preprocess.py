import json
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Load JSON dataset
with open("intents.json", "r") as file:
    data = json.load(file)

lemmatizer = WordNetLemmatizer()
tokenizer = Tokenizer()

tags = []
patterns = []
responses = {}

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        word_list = nltk.word_tokenize(pattern)
        patterns.append(" ".join([lemmatizer.lemmatize(word.lower()) for word in word_list]))
        tags.append(intent["tag"])

    responses[intent["tag"]] = intent["responses"]

# Encode labels
encoder = LabelEncoder()
encoded_tags = encoder.fit_transform(tags)

# Tokenize and pad sequences
tokenizer.fit_on_texts(patterns)
sequences = tokenizer.texts_to_sequences(patterns)
padded_sequences = pad_sequences(sequences, padding="post")

# Save encoders and tokenizer
import pickle
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

np.save("padded_sequences.npy", padded_sequences)
np.save("encoded_tags.npy", encoded_tags)

print("Data preprocessed and saved successfully!")
