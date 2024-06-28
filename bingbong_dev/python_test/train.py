import json
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

def preprocess_text(text):
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    import string
    import nltk

    nltk.download('stopwords')
    nltk.download('wordnet')

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    text = text.lower()  # Convert to lowercase
    text = ''.join([char for char in text if char.isalpha() or char.isspace()])  # Remove punctuation and numbers
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])  # Lemmatize and remove stopwords

    return text

# Load models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_directory = os.path.join(BASE_DIR, "saved_models")
svm_model_path = os.path.join(model_directory, 'best_svm_model.pkl')
svm_model = joblib.load(svm_model_path)

# New data for prediction
new_patterns = ["I am feeling really happy today!", "I lost my job and I am very sad"]
new_patterns_cleaned = [preprocess_text(pattern) for pattern in new_patterns]

# SVM Prediction
new_X = vectorizer.transform(new_patterns_cleaned).toarray()
svm_predictions = svm_model.predict(new_X)
svm_predicted_tags = label_encoder.inverse_transform(svm_predictions)

# Displaying responses
for pattern, tag in zip(new_patterns, svm_predicted_tags):
    response = df[df['tag'] == tag]['response'].values[0]
    print(f"Input: {pattern}\nPredicted Tag: {tag}\nResponse: {response}\n")

