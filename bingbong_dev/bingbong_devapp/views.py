from venv import logger
from django.shortcuts import render, redirect, get_object_or_404
from .models import *
from .forms import *
from django.contrib import messages
from django.db.models import Q
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .bingbong import *
from django.contrib.auth import logout
from django.contrib.auth.hashers import make_password, check_password
from datetime import datetime
import json
from python_test.train import generate_response


def land(request):
    return render(request, 'bingbong_devapp/landing.html')

def is_authenticated(request):
    return request.session.get('member') is not None

def get_session_data(request):
    return request.session.get('member', {})

def signin(request):
    if request.method == "GET":
        identifier = request.GET.get('identifier')
        password = request.GET.get('pass')
        now = datetime.now()
        iso_formatted_date = now.isoformat() 

        try:
            member = Member.objects.get(Q(email=identifier) | Q(username=identifier))
            if check_password(password, member.password):
                member.last_login = iso_formatted_date
                member.save(update_fields=['last_login'])
                request.session['member'] = {
                    'username': member.username,
                    'email': member.email,
                    'last_login': member.last_login
                }
                return redirect('bingbong')
            else:
                messages.error(request, 'Incorrect password')
        except Member.DoesNotExist:
            messages.error(request, 'User not found')

    return render(request, 'bingbong_devapp/login.html')

def signup(request):
    if request.method == "POST":
        form = Memberform(request.POST or None)

        if form.is_valid():
            
            password = request.POST.get('password')
            confirmpass = request.POST.get('confirmpassword')

            if password == confirmpass:
                member = form.save(commit=False)
                member.password = make_password(password)
                member.save()

                messages.success(request, 'Signup successful!')
                return redirect('login') 
            else:
                messages.error(request, 'Passwords do not match.')
        else:
            print("Form Not Valid!")
    else:
        return render(request, 'bingbong_devapp/signup.html')
    
def bingbong(request):
    member = request.session.get('member')
    username = member.get('username')
    context = {
        'member_username': username,
    }
    return render(request, 'bingbong_devapp/bingbong.html', context)

@csrf_exempt
def save_conversation(request):
    if request.method == 'POST':
        member_data = request.session.get('member')
        conversation_data_json = request.POST.get('conversation')
        conversationId = request.POST.get('conversationID')

        if member_data and conversation_data_json:
            try:
                member = Member.objects.get(username=member_data['username'])
                
                conversation_data = json.loads(conversation_data_json)
                print("Json:", conversation_data)
                
                if conversationId: 
                    conversation = Conversation.objects.get(conversation_id=conversationId) 
                    conversation.messages = json.dumps(conversation_data)
                    if not conversation.analyzed and len(conversation_data) <= 4:
                        result = analyze_conversation(conversation_data)
                        conversation.sentiment = result['average_sentiments']
                        conversation.analyzed = True
                else:
                    conversation = Conversation.objects.create(
                        member=member, 
                        messages=json.dumps(conversation_data)
                    )
                
                topic = f"Record {datetime.now().strftime("%m-%d-%Y %I:%M %p")}"
                conversation.topic = topic
                    
                conversation.save()

                return JsonResponse({'status': 'success', 'conversation_id': conversation.conversation_id})
            except (Member.DoesNotExist, json.JSONDecodeError, IndexError) as e: 
                return JsonResponse({'status': 'error', 'message': str(e)}, status=400) 
        else:
            return JsonResponse({'status': 'error', 'message': 'Missing data'}, status=400)
    else:
        return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=405)

@csrf_exempt
def get_conversations(request):
    if request.method == "GET":
        member_data = request.session.get('member')
        if member_data:
            try:
                member = Member.objects.get(username=member_data['username'])
                conversations = Conversation.objects.filter(member=member)
                serialized_convos = [
                    {'conversation_id': c.conversation_id, 'topic': c.topic, 'sentiment': c.sentiment}
                    for c in conversations
                ]
                return JsonResponse({'conversations': serialized_convos})
            except Exception as e:
                return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
        else:
            return JsonResponse({'status': 'error', 'message': 'Missing data'}, status=400)
    else:
        return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=405)

@csrf_exempt
def get_conversation_details(request):
    if request.method == "GET":
        conversation_id = request.GET.get('conversation_id')
        try:
            conversation = get_object_or_404(Conversation, conversation_id=conversation_id)
            messages = json.loads(conversation.messages)
            return JsonResponse({'messages': messages})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
    else:
        return JsonResponse({'status': 'error', 'message': 'Authentication required'}, status=401)

@csrf_exempt
def process_message(request):
    if request.method == 'POST':
        user_message = request.POST.get('message', '')
        if user_message.lower() == 'exit':
            return JsonResponse({'response': "Bot: Goodbye!"})

        bot_response = generate_response(user_message)
        return JsonResponse({'response': bot_response})
    else:
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    
def logout_view(request):
    logout(request)
    print("Logged out:", request.session.get('member'))
    return redirect('login')