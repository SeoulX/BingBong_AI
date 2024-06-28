from datetime import datetime
from django.shortcuts import render, redirect, get_object_or_404
from .models import *
from .forms import *
from django.contrib import messages
from django.db.models import Q
from django.http import JsonResponse
from functools import wraps
from django.db.models import Count

def signin(request):
    if request.method == "GET":
        identifier = request.GET.get('identifier')
        passw = request.GET.get('pass')
        now = datetime.now()
        hour = now.hour
        greeting = 'Good Morning' if 5 <= hour < 12 else ('Good afternoon' if 12 <= hour < 18 else 'Good Evening')
        try:
            member = Member.objects.get(Q(email=identifier) | Q(username=identifier), password=passw)
            request.session['member'] = {
                    'username': member.username,
                    'email': member.email,
                    'password' : member.password,
                }
        except Member.DoesNotExist as e:
            print(f"Error: {e}")
            return render(request, 'bingbong_devapp/login.html')
    else:
        return render(request, 'bingbong_devapp/lognin.html')

def signup(request):
    if request.method == "POST":
        form = Memberform(request.POST or None)
        if form.is_valid():
            member = form.save()

            password = request.POST.get('password')
            confirmpass = request.POST.get('confirmpass')

            if password == confirmpass:
                request.session['member'] = {
                    'username': member.username,
                    'email': member.email,
                }
            else:
                messages.error(request, 'Passwords do not match.')
                return redirect('login')
    else:
        return render(request, 'bingbong_devapp/signup.html')


    
