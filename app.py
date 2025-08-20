import streamlit as st
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load tokenizer from file, cached for performance
@st.cache_resource
def load_tokenizer():
    with open('tokenizer.pkl', 'rb') as handle:
        return pickle.load(handle)

# Load trained model, cached for performance
@st.cache_resource
def load_sentiment_model():
    return load_model('sentiment_model.h5')

# Load tokenizer and model
tokenizer = load_tokenizer()
model = load_sentiment_model()

# Define sentiment labels in the same order as factorize output
unique = ['positive', 'negative']  # Replace with your actual unique list if different

def predict_sentiment(text):
    tw = tokenizer.texts_to_sequences([text])
    tw = pad_sequences(tw, maxlen=200)
    prob = model.predict(tw)[0][0]          # Extract the scalar float
    prediction = int(round(prob))           # Now safe to round
    label = unique[prediction]
    st.write(f"Predicted label: {label}")
    return label

st.title("User Sentiment Analysis")
user_input = st.text_area("Enter a review to analyze sentiment:")

if st.button("Predict"):
    if user_input.strip():
        predict_sentiment(user_input)
    else:
        st.warning("Please enter a review text.")
