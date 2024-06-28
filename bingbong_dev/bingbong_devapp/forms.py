from django import forms
from .models import *

class Memberform(forms.ModelForm):
    class Meta:
        model = Member
        fields = ['username', 'email', 'password']