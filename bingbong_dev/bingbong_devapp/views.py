from django.shortcuts import render
from django.http import JsonResponse

# Initialize NLTK's VADER sentiment analyzer

def home(request):
    return render(request, 'bingbong_devapp/bingbong.html')

def process_message(request):
    print("drian")
