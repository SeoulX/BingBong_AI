from django.shortcuts import render, redirect, get_object_or_404
from .models import Member, Conversation
from .forms import Memberform
from django.contrib import messages
from django.db.models import Q
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .bingbong import get_response
from django.contrib.auth.hashers import make_password, check_password
import datetime


def land(request):
    return render(request, 'bingbong_devapp/bingbong.html')

def is_authenticated(request):
    return request.session.get('member') is not None

def get_session_data(request):
    return request.session.get('member', {})

def signin(request):
    if request.method == "POST":
        identifier = request.POST.get('identifier')
        password = request.POST.get('pass')

        try:
            member = Member.objects.get(Q(email=identifier) | Q(username=identifier))
            if member.check_password(password):  # Use check_password to verify
                request.session['member'] = {
                    'id': member.id,
                    'username': member.username,
                    'email': member.email,
                }
                return redirect('bingbong')
            else:
                messages.error(request, 'Incorrect password')
        except Member.DoesNotExist:
            messages.error(request, 'User not found')

    return render(request, 'bingbong_devapp/login.html')

def signup(request):
    if request.method == "POST":
        form = Memberform(request.POST)
        if form.is_valid():
            member = form.save(commit=False)
            member.password = make_password(member.password)  
            member.save()

            # Create initial conversation
            Conversation.objects.create(member=member, topic="General")

            request.session['member'] = {
                'id': member.id,
                'username': member.username,
                'email': member.email,
            }
            messages.success(request, 'Signup successful! You can now log in.')
            return redirect('login')  # Assuming 'login' is your login URL name
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = Memberform()

    return render(request, 'bingbong_devapp/signup.html', {'form': form})
    
def bingbong(request):
    if is_authenticated(request):
        print(is_authenticated(request))
        return render(request, 'bingbong_devapp/bingbong.html')
    else:
        return redirect('login')

@csrf_exempt
def process_message(request):
    if request.method == 'POST':
        user_message = request.POST.get('message', '')
        if user_message.lower() == 'exit':
            return JsonResponse({'response': "Bot: Goodbye!"})

        member_id = request.session['member']['id']
        conversation, created = Conversation.objects.get_or_create(
            member_id=member_id,
            topic="General"  # You might want to make this dynamic later
        )

        bot_response = get_response(user_message, conversation)
        return JsonResponse({'response': bot_response})
    else:
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    
