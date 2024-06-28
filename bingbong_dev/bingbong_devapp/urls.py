from django.urls import path
from . import views

urlpatterns = [
    path('', views.land, name='land'),
    path('login/', views.signin, name='login'),
    path('signup/', views.signup, name='signup'),
]
