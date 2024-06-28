from django.urls import path
from . import views

urlpatterns = [
    path('', views.land, name='land'),
    path('login/', views.signin, name='login'),
    path('signup/', views.signup, name='signup'),
    path('bingbong/', views.bingbong, name='bingbong'),
    
    # Processes
    path('process_message/', views.process_message, name='process_message'),
]
