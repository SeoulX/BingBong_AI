import json
import spacy
import os

nlp = spacy.load("en_core_web_sm")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
json_file_path = os.path.join(BASE_DIR, 'KB.json')

with open(json_file_path, 'r') as file:
    kb_data = json.load(file)

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
    """
    Analyzes sentiment and extracts a basic topic from conversation data,
    only if there are exactly two messages from the user.
    """

    user_messages = [msg["message"] for msg in conversation_data]

    if len(user_messages) != 10:
        return {"average_sentiments": None, "topic": None}

    sentiments = [sia.polarity_scores(msg)["compound"] for msg in user_messages]
    average_sentiment = sum(sentiments) / len(sentiments)

    all_words = []
    for message in user_messages:
        words = word_tokenize(message)
        words = [word.lower() for word in words if word.isalnum() and word.lower() not in stopwords.words('english')]
        all_words.extend(words)

    fdist = FreqDist(all_words)
    topic = fdist.max() if fdist else None

    return {
        "average_sentiments": average_sentiment,
        "topic": topic
    }

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print(BASE_DIR)