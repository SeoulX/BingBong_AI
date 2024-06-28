import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
import os

# 1. Load and Prepare Data from kb.json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get current file's directory
json_file_path = os.path.join(BASE_DIR, 'KB.json')

with open(json_file_path, 'r') as file:
    kb_data = json.load(file)

texts = []
labels = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        texts.append(pattern)
        labels.append(intent['tag'])

# 2. Tokenization and Padding

tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")  # Adjust num_words if needed
tokenizer.fit_on_texts(texts)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(texts)
padded = pad_sequences(sequences, padding='post', truncating='post')

# 3. Prepare Labels

label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(labels)
training_labels = np.array(label_tokenizer.texts_to_sequences(labels))

# 4. Build the Model

model = Sequential([
    Embedding(input_dim=len(word_index) + 1, output_dim=16),  # Adjust embedding dimensions
    GlobalAveragePooling1D(),
    Dense(16, activation='relu'), 
    Dense(len(label_tokenizer.word_index) + 1, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])

# 5. Train the Model

model.fit(padded, training_labels, epochs=50, verbose=1)  # Adjust epochs as needed

# 6. Save the Model (Optional)

model.save('chatbot_model.h5')
