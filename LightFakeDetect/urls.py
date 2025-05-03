"""
URL configuration for LightFakeDetect project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
"""

from django.urls import path
from . import views

urlpatterns = [
    # Use views we know exist based on your views.py file
    path("", views.model_comparison_view, name="news_home"),
    path("models/comparison/", views.model_comparison_view, name="model_comparison"),
    path("analyze/text/", views.analyze_text_view, name="analyze_text"),
]