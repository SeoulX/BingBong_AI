from django.db import models
import uuid

from django.forms import JSONField

class Member(models.Model):
    username = models.CharField(max_length=30)
    email = models.EmailField(max_length=30)
    password = models.CharField(max_length=16)
    active = models.BooleanField(default=True) 
    last_login = models.DateTimeField(null=True, blank=True)
    
    def __str__(self):
        return self.username
class Conversation(models.Model):
    conversation_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    member = models.ForeignKey(Member, on_delete=models.CASCADE, related_name='member_convo')
    topic = models.TextField()
    sentiment = models.TextField(null=True)
    messages = models.JSONField(default=list)
    analyzed = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.topic