"""
@file news/views.py
@brief Views for displaying news articles and related content.

This module provides view functions for displaying news articles,
filtering by topics, and showing bias analysis results.

@author Ameed Othman
@date 2025-04-02
"""

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import get_object_or_404, redirect, render

from .models import Article, DetectionModelMetrics, Source
from .services import analyze_article, detect_fake_news


def home_view(request):
    """
    View for homepage displaying recent news articles.
    """
    # Get all sources for filter dropdown
    sources = Source.objects.all()

    # Get latest articles, limited to 12 for performance
    articles = Article.objects.select_related('source').order_by('-publication_date')[:12]

    context = {
        'articles': articles,
        'sources': sources,
    }

    return render(request, 'news/home.html', context)


def article_detail_view(request, article_id):
    """
    View for displaying a single article with detection results.
    """
    article = get_object_or_404(Article, id=article_id)

    # Increment view count
    article.view_count += 1
    article.save(update_fields=['view_count'])

    # Get related articles (same source)
    related_articles = Article.objects.filter(source=article.source).exclude(id=article_id).order_by(
        '-publication_date')[:5]

    context = {
        'article': article,
        'related_articles': related_articles,
    }

    return render(request, 'news/article_detail.html', context)


def source_detail_view(request, source_id):
    """
    View for displaying articles from a specific source.
    """
    source = get_object_or_404(Source, id=source_id)

    # Get articles from this source
    articles = Article.objects.filter(source=source).order_by('-publication_date')[:12]

    context = {
        'source': source,
        'articles': articles,
    }

    return render(request, 'news/source_detail.html', context)


def search_view(request):
    """
    View for searching articles by keyword.
    """
    query = request.GET.get('q', '')

    if query:
        # Search in title and content
        articles = Article.objects.filter(
            title__icontains=query
        ).order_by('-publication_date')[:20]
    else:
        articles = []

    context = {
        'query': query,
        'articles': articles,
        'total_results': len(articles),
    }

    return render(request, 'news/search_results.html', context)


def model_comparison_view(request):
    """
    View for comparing different detection model performance.
    THIS IS A KEY VIEW FOR YOUR GRADUATION PROJECT.
    """
    # Get metrics for all models
    model_metrics = DetectionModelMetrics.objects.all().order_by('-f1_score')

    context = {
        'model_metrics': model_metrics,
    }

    return render(request, 'news/model_comparison.html', context)


def analyze_text_view(request):
    """
    View for analyzing custom text with different models.
    THIS IS A KEY VIEW FOR YOUR GRADUATION PROJECT DEMO.
    """
    results = None

    if request.method == 'POST':
        text = request.POST.get('text', '')
        model_key = request.POST.get('model', 'distilbert')

        if text:
            # Analyze the text
            results = detect_fake_news(text, model_key)

    # Always provide model options
    models = [
        {'key': 'distilbert', 'name': 'DistilBERT'},
        {'key': 'tinybert', 'name': 'TinyBERT'},
    ]

    context = {
        'results': results,
        'models': models,
    }

    return render(request, 'news/analyze_text.html', context)


@login_required
def analyze_article_view(request, article_id):
    """
    View for manually triggering article analysis.
    """
    article = get_object_or_404(Article, id=article_id)
    model_key = request.POST.get('model', 'distilbert')

    # Analyze the article
    success = analyze_article(article.id, model_key)

    if success:
        messages.success(request, f'Article "{article.title[:30]}..." analyzed successfully.')
    else:
        messages.error(request, 'Failed to analyze article. Please try again.')

    # Redirect back to article
    return redirect('article_detail', article_id=article_id)