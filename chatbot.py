import numpy as np
import json
import pickle
import nltk
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import google.generativeai as genai
import os

# Load necessary files
with open("intents.json", "r") as file:
    data = json.load(file)
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
with open("label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# Load trained model
model = load_model("chatbot_lstm.h5")

# Preprocessing function
lemmatizer = nltk.stem.WordNetLemmatizer()
def preprocess_input(text):
    word_list = nltk.word_tokenize(text)
    word_list = [lemmatizer.lemmatize(word.lower()) for word in word_list]
    sequence = tokenizer.texts_to_sequences([" ".join(word_list)])
    return pad_sequences(sequence, maxlen=10, padding="post")

# Function to predict intent
def predict_intent(user_input):
    processed_input = preprocess_input(user_input)
    response = model.generate_content(str(processed_input))
    prediction = response.text  # Extract text from response

    tag = encoder.inverse_transform([np.argmax(prediction)])[0]
    return tag

genai.configure(api_key="AIzaSyBMLfz9o-XtoqJAE71nr1JfWXznZs6rDJ4")

model = genai.GenerativeModel("gemini-2.0-flash")

def get_gemini_response(user_input):
    response = model.generate_content(user_input)
    return response.text 

# Function to return response based on intent
def get_response(intent, user_input):
    responses = {
        "greeting": ["Hello! How can I assist?", "Hi there! Need help?", "Hey! What's up?"],
        "goodbye": ["Goodbye! Have a great day!", "See you later!", "Take care!"],
        "joke": ["Why did the chicken cross the road? To get to the other side! 😂"],
    }
    
    if intent in responses:
        return np.random.choice(responses[intent])
    else:
        return get_gemini_response(user_input)  # GPT-4 response for unknown intents

# Chatbot function
def chatbot():
    print("anarbot is running! (Type 'quit' to exit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            print("Goodbye! 👋")
            break
        intent = predict_intent(user_input)
        response = get_response(intent, user_input)
        print("anarbot:", response)

# Run chatbot
chatbot()
