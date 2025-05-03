from django.urls import path
from . import views

urlpatterns = [
    # Home/index page
    path("", views.model_comparison_view, name="news_home"),

    # Model comparison page
    path("models/comparison/", views.model_comparison_view, name="model_comparison"),

    # Text analysis page
    path("analyze/text/", views.analyze_text_view, name="analyze_text"),
]