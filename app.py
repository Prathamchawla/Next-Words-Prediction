import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from PIL import Image

# Load models and tokenizer
@st.cache_resource
def load_models():
    lstm_model = tf.keras.models.load_model('LSTM_WordPrediction.h5')
    gru_model = tf.keras.models.load_model('GRU_WordPrediction.h5')
    return lstm_model, gru_model

lstm_model, gru_model = load_models()

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

# Prediction function
def predict_next_words(model, tokenizer, text, max_sequence_len, num_words=3):
    for _ in range(num_words):
        token_list = tokenizer.texts_to_sequences([text])[0]
        if len(token_list) >= max_sequence_len:
            token_list = token_list[-(max_sequence_len-1):]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        predicted_word_index = np.argmax(predicted, axis=1)[0]
        predicted_word = None
        for word, index in tokenizer.word_index.items():
            if index == predicted_word_index:
                predicted_word = word
                break
        if not predicted_word:
            break
        text += " " + predicted_word
    return text

# Set up Streamlit interface
st.title("Next Words Prediction with LSTM and GRU")
st.write("Choose a model, enter some initial text, and predict the next words.")

# Sidebar for image display
with st.sidebar:
    st.write("### Training and Validation Loss Plots")
    lstm_loss_image = Image.open("LSTMtraining_validation_loss.png")  # Replace with the actual filename
    gru_loss_image = Image.open("GRUtraining_validation_loss.png")    # Replace with the actual filename
    bleu_score_image = Image.open("bleu_score_comparison.png")        # Replace with the actual filename

    st.image(lstm_loss_image, caption="LSTM Model Loss")
    st.image(gru_loss_image, caption="GRU Model Loss")
    st.image(bleu_score_image, caption="BLEU Score Comparison")

# Text input
input_text = st.text_input("Enter your text:", "Hello")

# Model selection
model_option = st.selectbox("Choose a model:", ("LSTM", "GRU"))
num_words = st.slider("Number of words to predict:", min_value=1, max_value=10, value=3)

# Predict next words
if st.button("Predict Next Words"):
    max_sequence_len = lstm_model.input_shape[1] + 1
    model = lstm_model if model_option == "LSTM" else gru_model
    result_text = predict_next_words(model, tokenizer, input_text, max_sequence_len, num_words)
    st.write("Predicted text:", result_text)
