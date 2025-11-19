from django.contrib import admin
from django.urls import path
from app import views

urlpatterns = [
    path('',views.index,name='index'),
    path('login', views.user_login , name='login'), 
    path('logout/', views.log_out, name='log_out'), 
    path('register', views.register , name='register'), 
    path('dashboard', views.dashboard , name='dashboard'),
    path('chatbot-response/', views.chatbot_response, name='chatbot_response'),
    path('addpklfiles', views.addpklfiles , name='addpklfiles'),
    path('start_exercise/', views.start_exercise, name='start_exercise'),
    path("get-profile", views.get_profile, name="get_profile"),
    path("get_exercises_last_five_days/", views.get_exercises_last_five_days, name="get_exercises_last_five_days"),
]
