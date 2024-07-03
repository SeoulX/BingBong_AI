from django import forms
from .models import *
from django.contrib.auth.hashers import make_password, check_password

class Memberform(forms.ModelForm):
    confirm_password = forms.CharField(widget=forms.PasswordInput)  

    class Meta:
        model = Member
        fields = ['username', 'email', 'password']
        widgets = {
            'password': forms.PasswordInput(),
        }

    def clean(self):
        cleaned_data = super().clean()
        password = cleaned_data.get('password')
        confirm_password = cleaned_data.get('confirm_password')

        if password and confirm_password and password != confirm_password:
            raise forms.ValidationError("Passwords do not match.")

    def save(self, commit=True):
        member = super().save(commit=False)
        member.password = make_password(member.password) 
        if commit:
            member.save()
        return member