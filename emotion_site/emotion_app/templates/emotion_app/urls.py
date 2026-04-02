from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('', views.landing, name='home'),
    path('analyze/', views.analyze, name='analyze'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('about/', views.about, name='about'),
]