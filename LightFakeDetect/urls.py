"""
URL configuration for LightFakeDetect project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
"""

from django.contrib import admin
from django.urls import include, path
from django.views.generic import RedirectView

urlpatterns = [
    path("admin/", admin.site.urls),
    path("models/", include("news.urls")),
    path("", RedirectView.as_view(pattern_name="model_comparison"), name="home"),
]