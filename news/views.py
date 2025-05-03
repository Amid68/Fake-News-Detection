"""
@file news/views.py
@brief Views for displaying news articles and related content.

This module provides view functions for displaying news articles,
filtering by topics, and showing bias analysis results.

@author Ameed Othman
@date 2025-04-02
"""

from django.shortcuts import render
from .services import detect_fake_news, MODELS
from .models import DetectionModelMetrics


def model_comparison_view(request):
    """View for comparing different detection model performance"""
    model_metrics = DetectionModelMetrics.objects.all().order_by('-f1_score')

    return render(request, 'news/model_comparison.html', {
        'model_metrics': model_metrics
    })


def analyze_text_view(request):
    """View for analyzing custom text with different models"""
    results = None

    if request.method == 'POST':
        text = request.POST.get('text', '')
        model_key = request.POST.get('model', 'distilbert')

        if text:
            results = detect_fake_news(text, model_key)

    # Model options for the form
    models = [{'key': key, 'name': config['name']} for key, config in MODELS.items()]

    return render(request, 'news/analyze_text.html', {
        'results': results,
        'models': models
    })

def examples_view(request):
    """View for displaying example news articles for testing"""
    return render(request, 'news/examples.html')