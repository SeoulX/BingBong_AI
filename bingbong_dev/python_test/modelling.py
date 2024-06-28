import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, TimeDistributed
from tensorflow.keras.models import Sequential
import os

# 1. Load and Prepare Data from KB.json
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, 'KB.json')

def load_dataset(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    inputs, responses = [], []
    for intent in data['intents']:
        if 'responses' in intent and intent['responses']: 
            for pattern in intent['patterns']:
                inputs.append(pattern)
                responses.append(intent['responses'][0])  
        else:
            print(f"Warning: Intent '{intent['tag']}' has no responses defined.")
    return inputs, responses

# Load data before model preparation
inputs, responses = load_dataset(file_path)

# 2. Prepare Data for Model (Only if data was loaded)
if inputs and responses:
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
    
    # 3. Build & Compile the Model
    model = Sequential([
        Embedding(vocab_size, embedding_dim),
        LSTM(128, return_sequences=True),
        LSTM(128, return_sequences=True),   # Note: return_sequences=True for both LSTMs
        TimeDistributed(Dense(vocab_size, activation='softmax')),           # Added TimeDistributed wrapper
    ])
    
    # use sparse_categorical_crossentropy loss
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Data Generator
    def data_generator(padded_inputs, padded_outputs, batch_size):
        while True:
            for i in range(0, len(padded_inputs), batch_size):
                yield padded_inputs[i:i+batch_size], padded_outputs[i:i+batch_size] 
        
    # Train the model
    batch_size = 32
    steps_per_epoch = len(padded_inputs) // batch_size
    model.fit(data_generator(padded_inputs, padded_outputs, batch_size), epochs=10, steps_per_epoch=steps_per_epoch)
        
    # Train the model
    batch_size = 32
    steps_per_epoch = len(padded_inputs) // batch_size
    model.fit(data_generator(padded_inputs, padded_outputs, batch_size), epochs=10, steps_per_epoch=steps_per_epoch)
        
    # 4. Chat Interaction Loop
    def chat():
        print("Bot: Hi there! How are you feeling today?")
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'quit':
                break

            input_seq = tokenizer.texts_to_sequences([user_input])
            padded_seq = pad_sequences(input_seq, maxlen=max_length, padding='post')
            predicted_seq = model.predict(padded_seq)[0]

            # Find the word index with the highest probability for each time step
            predicted_word_index = np.argmax(predicted_seq, axis=-1)
            
            # Reverse word index to get the word from predicted word index
            reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
            decoded_text = " ".join([reverse_word_index.get(i, "?") for i in predicted_word_index])

            print("Bot:", decoded_text)
else:
    print("Error: No valid data loaded from kb.json. Please check the file format.")

chat()
