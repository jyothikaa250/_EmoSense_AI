from django.urls import path
from . import views

urlpatterns = [
    path('', views.landing, name='landing'),
    path('analyze/', views.analyze, name='analyze'),
    path('about/', views.about, name='about'),
    path('dashboard/', views.dashboard, name='dashboard'),
]