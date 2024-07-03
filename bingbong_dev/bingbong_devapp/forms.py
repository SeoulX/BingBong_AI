from django import forms
from .models import *
from django.contrib.auth.hashers import make_password, check_password

class Memberform(forms.ModelForm):
    class Meta:
        model = Member
        fields = ['username', 'email', 'password']