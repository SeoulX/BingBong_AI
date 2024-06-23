from django.shortcuts import render
from django.http import JsonResponse
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize NLTK's VADER sentiment analyzer
nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()

def home(request):
    return render(request, 'chatbot/home.html')

def process_message(request):
    if request.method == 'POST':
        user_message = request.POST.get('message', '')
        
        # Perform sentiment analysis
        sentiment_scores = sid.polarity_scores(user_message)
        compound_score = sentiment_scores['compound']
        
        # Determine response based on sentiment
        if compound_score >= 0.05:
            response = "I'm glad you're feeling okay!"
        elif compound_score > -0.05 and compound_score < 0.05:
            response = "It sounds like you're feeling neutral."
        else:
            response = "I'm sorry you're feeling this way. Can you tell me more about it?"
        
        return JsonResponse({'response': response})
    else:
        return JsonResponse({'error': 'Invalid request method'})
