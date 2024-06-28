import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, TimeDistributed
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
import os

# 1. Load and Prepare Data from KB.json
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, 'KB.json')

# 1. Load and Prepare Data
def load_dataset(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    inputs, responses = [], []
    for intent in data['intents']:
        if 'responses' in intent and intent['responses']:
            for pattern in intent['patterns']:
                inputs.append(pattern)
                responses.append(intent['responses'][0])
    return inputs, responses

# Load data before model preparation
inputs, responses = load_dataset(file_path)

# 2. Prepare Data for Model
vocab_size = 10000
embedding_dim = 128
max_length = 20
trunc_type = 'post'
oov_token = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(inputs + responses)
word_index = tokenizer.word_index

input_sequences = tokenizer.texts_to_sequences(inputs)
padded_inputs = pad_sequences(input_sequences, maxlen=max_length, truncating=trunc_type)
output_sequences = tokenizer.texts_to_sequences(responses)
padded_outputs = pad_sequences(output_sequences, maxlen=max_length, truncating=trunc_type)

# 3. Build Model
model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_length),
    Bidirectional(LSTM(128, return_sequences=True)),  # Bidirectional LSTM
    Dropout(0.2),  # Dropout for regularization
    LSTM(128, return_sequences=True),
    Dropout(0.2),
    TimeDistributed(Dense(vocab_size, activation='softmax'))
])

# Compile Model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Implement Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3)  # Add early stopping

# 4. Train Model (with Validation Split)
history = model.fit(padded_inputs, padded_outputs, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping])  # Validation split

# 5. Chat Loop
def chat():
    print("Bot: Hi there! How are you feeling today?")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break

        input_seq = tokenizer.texts_to_sequences([user_input])
        padded_seq = pad_sequences(input_seq, maxlen=max_length, padding='post')
        predicted_seq = model.predict(padded_seq)[0]
        predicted_word_index = np.argmax(predicted_seq, axis=-1)

        reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
        decoded_text = " ".join([reverse_word_index.get(i, "?") for i in predicted_word_index])
        print("Bot:", decoded_text)

# Load or train the model
model_path = 'my_chatbot_model.h5'
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
else:
    chat() 
