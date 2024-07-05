
import os
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict

sia = SentimentIntensityAnalyzer()
nltk.download('stopwords')
nltk.download('punkt')

def analyze_conversation(conversation_data):
    
    user_messages = [msg["message"] for msg in conversation_data if msg["sender"] != "BingBong"]
    
    print("User_messages:", user_messages)

    if len(user_messages) < 3:
        return {"average_sentiments": None}

    sentiments = [sia.polarity_scores(msg)["compound"] for msg in user_messages]
    average_sentiment = sum(sentiments) / len(sentiments)

    return {
        "average_sentiments": average_sentiment,
    }