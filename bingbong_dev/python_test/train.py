import json
import joblib
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import google.generativeai as genai
from dotenv import load_dotenv, dotenv_values 
load_dotenv()

genai.configure(api_key=os.getenv("API_KEY"))


# Download necessary nltk resources if not already downloaded
nltk.download('stopwords')
nltk.download('wordnet')

# Function to preprocess text
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = ''.join([char for char in text if char.isalpha() or char == ' '])
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    return text

# Load the saved models and data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_directory = os.path.join(BASE_DIR, "saved_models")

svm_model = joblib.load(os.path.join(model_directory, 'best_svm_model.pkl'))
vectorizer = joblib.load(os.path.join(model_directory, 'tfidf_vectorizer.pkl'))
label_encoder = joblib.load(os.path.join(model_directory, 'label_encoder.pkl'))

# Load the KB.json file
file_path = os.path.join(BASE_DIR, 'KB.json')
with open(file_path, 'r') as file:
    data = json.load(file)

# Function to calculate word-by-word cosine similarity
def calculate_similarity(text1, text2):
    vectorizer = CountVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    return cosine_similarity([vectors[0]], [vectors[1]])[0][0]

# Function to generate response using enhanced matching
def generate_response(user_input):
    # Preprocess user input
    user_input_cleaned = preprocess_text(user_input)
        
    # Initialize variables for best match
    best_response = "I'm not sure how to respond to that."
    max_similarity = 0.0
    
    try:
        # Iterate over intents in KB.json
        for intent in data['intents']:
            tag = intent['tag']
            patterns = intent['patterns']
            responses = intent['responses']
            
            # Compare user input against each pattern
            for pattern in patterns:
                pattern_cleaned = preprocess_text(pattern)
                
                # Calculate similarity between user input and pattern
                similarity_score = calculate_similarity(user_input_cleaned, pattern_cleaned)
                
                # Update best response if similarity score is higher
                if similarity_score > max_similarity:
                    max_similarity = similarity_score
                    best_response = np.random.choice(responses)
    except KeyError as e:
        print(f"KeyError: Missing key in intents: {e}")
        
    generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
    }

    model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    # safety_settings = Adjust safety settings
    # See https://ai.google.dev/gemini-api/docs/safety-settings
    )
    chat_session = model.start_chat(
        history=[
        ]
        )
    
    promp = f"User Input: {user_input} \n Model Input: {best_response} \n Give a humanized version of the model output. Always remember you are BingBong a friendly bot."
    
    response = chat_session.send_message(promp)
    
    return response.text

