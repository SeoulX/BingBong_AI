from django.db import models

class Member(models.Model):
    username = models.CharField(max_length=30)
    email = models.EmailField(max_length=30)
    password = models.CharField(max_length=16)
    active = models.BooleanField(default=True) 
    
    def __str__(self):
        return self.username
class Conversation(models.Model):
    member = models.ForeignKey(Member, on_delete=models.CASCADE, related_name='conversations')
    topic = models.TextField()
    context = models.TextField() 

    def __str__(self):
        return f"Conversation by {self.member.username}"