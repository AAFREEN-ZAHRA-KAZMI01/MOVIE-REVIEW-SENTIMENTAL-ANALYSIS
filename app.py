# app.py

import streamlit as st
import joblib
import re
import nltk
import numpy as np
from nltk.corpus import stopwords

# NLTK stopwords download karo
nltk.download('stopwords')

# Preprocessing function
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-z\s]', '', text)  # Remove special characters and numbers
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Load the trained model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Streamlit app UI
st.title("🎬 Movie Review Sentiment Analyzer")
st.write("Enter a movie review to predict whether it's positive or negative.")

# User input area
user_input = st.text_area("Enter your movie review:")

# Predict button
if st.button("Predict"):
    if user_input.strip() != "":
        processed_input = preprocess(user_input)
        vectorized_input = vectorizer.transform([processed_input])

        # Predict sentiment and confidence
        pred_prob = model.predict_proba(vectorized_input)[0]
        pred_label = np.argmax(pred_prob)
        confidence = np.max(pred_prob)

        if pred_label == 1:
            st.success(f"Prediction: Positive 😊 with {confidence*100:.2f}% confidence")
        else:
            st.error(f"Prediction: Negative 😞 with {confidence*100:.2f}% confidence")
    else:
        st.warning("Please enter a review.")
