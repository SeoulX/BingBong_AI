
import os
from string import punctuation
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict

nltk.download('punkt')
nltk.download('stopwords')

sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

# def analyze_conversation(conversation_data):
    
#     user_messages = [msg["message"] for msg in conversation_data if msg["sender"] != "BingBong"]
    
#     print("User_messages:", user_messages)

#     if len(user_messages) < 3:
#         return {"average_sentiments": None}

#     sentiments = [sia.polarity_scores(msg)["compound"] for msg in user_messages]
#     average_sentiment = sum(sentiments) / len(sentiments)
    
#     return {
#         "average_sentiments": average_sentiment,
#     }
    

def preprocess_text(message):
    words = word_tokenize(message.lower())
    
    filtered_words = [word for word in words if word not in stop_words and word not in punctuation]
    
    return " ".join(filtered_words)

def analyze_conversation(conversation_data):
    user_messages = [msg["message"] for msg in conversation_data if msg["sender"] != "BingBong"]
    
    print("User_messages:", user_messages)

    if len(user_messages) < 3:
        return {"average_sentiments": None}

    preprocessed_messages = [preprocess_text(user_messages[1])]
    print(preprocessed_messages)
    sentiments = [sia.polarity_scores(msg)["compound"] for msg in preprocessed_messages]
    average_sentiment = sum(sentiments) / len(sentiments)
    
    return {
        "average_sentiments": average_sentiment,
    }