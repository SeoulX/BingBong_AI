from django.urls import path
from . import views

urlpatterns = [
    path('', views.land, name='land'),
    path('login/', views.signin, name='login'),
    path('signup/', views.signup, name='signup'),
    path('bingbong/', views.bingbong, name='bingbong'),
    
    path('process_message/', views.process_message, name='process_message'),
    path('save_conversation/', views.save_conversation, name='save_conversation'),
]
