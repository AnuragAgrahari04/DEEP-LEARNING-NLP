import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for (key, value) in word_index.items()}

# load the pre-trained model with ReLu Activation
model = load_model('simplernn_imdb_model.h5')

#Step 2 : Helper function
# function to decode reviews

def decode_review(decoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in decoded_review])

# function to preprocess user input
def preprocess_review(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


#Step:3 prediction function
def predict_sentiment(review):
    preprocessed_input = preprocess_review(review)
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] >= 0.5 else 'Negative'
    return sentiment, prediction[0][0]

# streamlit app

import streamlit as st
st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review below to classify it as positive or negative.")

user_input = st.text_area("Movie Review")
if st.button("Classify"):

    preprocessed_input = preprocess_review(user_input)
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] >= 0.5  else 'Negative'

    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {prediction[0][0]}')
else:
    st.write('Please ender a movie review.')