import json
import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import nltk
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from tensorflow.keras.utils import to_categorical

nltk.download('stopwords')
nltk.download('wordnet')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, 'KB.json')

with open(file_path, 'r') as file:
    data = json.load(file)

tags = []
patterns = []
responses = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)
        if 'responses' in intent:
            responses.append(intent['responses'][0])
        else:
            responses.append(None)

df = pd.DataFrame({'tag': tags, 'pattern': patterns, 'response': responses})

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char.isalpha() or char == ' '])
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    return text

# Clean text data
df['pattern_cleaned'] = df['pattern'].apply(preprocess_text)
df['response_cleaned'] = df['response'].apply(lambda x: preprocess_text(x) if x else "")

# Fit and save TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['pattern_cleaned']).toarray()
joblib.dump(vectorizer, os.path.join(BASE_DIR, 'saved_models', 'tfidf_vectorizer.pkl'))

# Encode labels
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['tag'])
joblib.dump(label_encoder, os.path.join(BASE_DIR, 'saved_models', 'label_encoder.pkl'))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize models
nb_model = MultinomialNB()
svm_model = SVC(kernel='linear')
rf_model = RandomForestClassifier()

# Train models
nb_model.fit(X_train, y_train)
svm_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# Grid search for SVM model
param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
grid_search = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
grid_search.fit(X_train, y_train)
svm_best = grid_search.best_estimator_

# Cross-validation scores
nb_cv_scores = cross_val_score(nb_model, X, y, cv=5)
svm_cv_scores = cross_val_score(svm_best, X, y, cv=5)
rf_cv_scores = cross_val_score(rf_model, X, y, cv=5)

# Evaluation function
def evaluate_model(model, X_test, y_test, y_pred):
    unique_labels = np.unique(np.concatenate((y_test, y_pred)))
    target_names = label_encoder.inverse_transform(unique_labels)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

# Evaluate models
print("Naive Bayes Model")
nb_pred = nb_model.predict(X_test)
evaluate_model(nb_model, X_test, y_test, nb_pred)
print("Cross-validation scores:", nb_cv_scores)
print("Mean CV score:", np.mean(nb_cv_scores))

print("\nSVM Model")
svm_pred = svm_model.predict(X_test)
evaluate_model(svm_model, X_test, y_test, svm_pred)
print("Cross-validation scores:", svm_cv_scores)
print("Mean CV score:", np.mean(svm_cv_scores))

print("\nRandom Forest Model")
rf_pred = rf_model.predict(X_test)
evaluate_model(rf_model, X_test, y_test, rf_pred)
print("Cross-validation scores:", rf_cv_scores)
print("Mean CV score:", np.mean(rf_cv_scores))

print("\nBest SVM Model (Grid Search)")
svm_best_pred = svm_best.predict(X_test)
evaluate_model(svm_best, X_test, y_test, svm_best_pred)

# Plot confusion matrix for best SVM model
conf_matrix = confusion_matrix(y_test, svm_best_pred)
plt.figure(figsize=(14, 10))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.inverse_transform(np.unique(y_test)), yticklabels=label_encoder.inverse_transform(np.unique(y_test)))
plt.title('Confusion Matrix for Best SVM Model')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.show()

# Create word cloud
all_patterns = ' '.join(df['pattern_cleaned'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_patterns)
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Most Common Words in Patterns')
plt.show()

# LSTM Model parameters
MAX_NB_WORDS = 5000
MAX_SEQUENCE_LENGTH = 250
EMBEDDING_DIM = 100

# Tokenize and pad sequences for LSTM
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(df['pattern_cleaned'].values)
word_index = tokenizer.word_index
X_lstm = tokenizer.texts_to_sequences(df['pattern_cleaned'].values)
X_lstm = pad_sequences(X_lstm, maxlen=MAX_SEQUENCE_LENGTH)
y_lstm = pd.get_dummies(df['tag']).values

# Train-test split for LSTM
X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(X_lstm, y_lstm, test_size=0.2, random_state=42)

# Build LSTM model
model_lstm = Sequential()
model_lstm.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
model_lstm.add(SpatialDropout1D(0.2))
model_lstm.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model_lstm.add(Dense(len(y_lstm[0]), activation='softmax'))
model_lstm.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train LSTM model
history = model_lstm.fit(X_train_lstm, y_train_lstm, epochs=5, batch_size=64, validation_data=(X_test_lstm, y_test_lstm), verbose=2)

# Plot training history of LSTM model
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluate LSTM model
y_test_lstm_pred = model_lstm.predict(X_test_lstm)
y_test_lstm_pred_class = np.argmax(y_test_lstm_pred, axis=1)
y_test_lstm_class = np.argmax(y_test_lstm, axis=1)
print("LSTM Model")
evaluate_model(model_lstm, X_test_lstm, y_test_lstm_class, y_test_lstm_pred_class)

# Save models
model_directory = os.path.join(BASE_DIR, "saved_models")
if not os.path.exists(model_directory):
    os.makedirs(model_directory)

joblib.dump(svm_best, os.path.join(model_directory, 'best_svm_model.pkl'))
model_lstm.save(os.path.join(model_directory, 'lstm_model.h5'))
