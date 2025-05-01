"""
@file news/views.py
@brief Views for displaying news articles and related content.

This module provides view functions for displaying news articles,
filtering by topics, and showing bias analysis results.

@author Ameed Othman
@date 2025-04-02
"""

from datetime import timedelta

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.db.models import Q
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone

from users.models import UserPreference

from .models import (Article, ArticleViewHistory, DetectionModelMetrics,
                     Source, Topic, UserSavedArticle)


def home_view(request):
    """
    View function for the home page displaying recent news articles.
    """
    # Get all sources for filter dropdown
    sources = Source.objects.filter(is_active=True).order_by("name")

    # Get all topics for filter dropdown
    topics = Topic.objects.all().order_by("name")

    # Base queryset for articles
    articles = Article.objects.select_related("source").order_by("-publication_date")

    # Apply filters if provided
    source_id = request.GET.get("source")
    topic_slug = request.GET.get("topic")
    days_str = request.GET.get("days")

    # Filter by source if specified
    if source_id:
        try:
            articles = articles.filter(source_id=int(source_id))
        except ValueError:
            pass  # Invalid source ID

    # Filter by topic if specified
    if topic_slug:
        articles = articles.filter(topics__slug=topic_slug)

    # Filter by date range if specified
    if days_str:
        try:
            days = int(days_str)
            date_threshold = timezone.now() - timedelta(days=days)
            articles = articles.filter(publication_date__gte=date_threshold)
        except ValueError:
            pass  # Invalid days value

    # Personalize for authenticated users if no specific filters
    if request.user.is_authenticated and not (source_id or topic_slug):
        # Get user preferences
        preferences = UserPreference.objects.filter(user=request.user)
        if preferences.exists():
            # Create a query filter based on preferences
            preference_query = Q()
            for pref in preferences:
                keyword = pref.topic_keyword.lower()
                preference_query |= Q(title__icontains=keyword) | Q(
                    content__icontains=keyword
                )

            # Filter articles by preferences if any exist
            if preference_query:
                articles = articles.filter(preference_query)

    # Paginate the results
    paginator = Paginator(articles, 12)  # Show 12 articles per page
    page_number = request.GET.get("page", 1)
    page_obj = paginator.get_page(page_number)

    context = {
        "page_obj": page_obj,
        "sources": sources,
        "topics": topics,
        "current_source": source_id,
        "current_topic": topic_slug,
        "current_days": days_str,
    }

    return render(request, "news/home.html", context)


def article_detail_view(request, article_id):
    """
    View function for displaying a single article with its details.
    """
    article = get_object_or_404(Article, id=article_id)

    # Increment view count
    article.view_count += 1
    article.save(update_fields=["view_count"])

    # Record view history for authenticated users
    if request.user.is_authenticated:
        ArticleViewHistory.objects.create(user=request.user, article=article)

    # Check if the article is saved by the current user
    is_saved = False
    if request.user.is_authenticated:
        is_saved = UserSavedArticle.objects.filter(
            user=request.user, article=article
        ).exists()

    # Get related articles (same topics or source)
    related_articles = (
        Article.objects.select_related("source")
        .filter(Q(topics__in=article.topics.all()) | Q(source=article.source))
        .exclude(id=article_id)
        .distinct()
        .order_by("-publication_date")[:5]
    )

    context = {
        "article": article,
        "is_saved": is_saved,
        "related_articles": related_articles,
    }

    return render(request, "news/article_detail.html", context)


@login_required
def save_article_view(request, article_id):
    """
    View function for saving an article for later reading.
    """
    article = get_object_or_404(Article, id=article_id)

    # Create saved article if it doesn't exist
    saved, created = UserSavedArticle.objects.get_or_create(
        user=request.user, article=article
    )

    if created:
        messages.success(
            request, f'Article "{article.title[:50]}..." saved for later reading.'
        )
    else:
        messages.info(request, "This article was already in your saved list.")

    # Get the referring page to redirect back
    next_url = request.GET.get("next", None)
    if next_url:
        return redirect(next_url)

    # Default redirect to article detail
    return redirect("article_detail", article_id=article_id)


@login_required
def unsave_article_view(request, article_id):
    """
    View function for removing an article from saved list.
    """
    article = get_object_or_404(Article, id=article_id)

    # Delete the saved article if it exists
    deleted, _ = UserSavedArticle.objects.filter(
        user=request.user, article=article
    ).delete()

    if deleted:
        messages.success(
            request, f'Article "{article.title[:50]}..." removed from your saved list.'
        )
    else:
        messages.info(request, "This article was not in your saved list.")

    # Get the referring page to redirect back
    next_url = request.GET.get("next", None)
    if next_url:
        return redirect(next_url)

    # Default redirect to article detail
    return redirect("article_detail", article_id=article_id)


@login_required
def saved_articles_view(request):
    """
    View function for displaying all articles saved by the user.
    """
    saved_articles = (
        UserSavedArticle.objects.filter(user=request.user)
        .select_related("article", "article__source")
        .order_by("-saved_at")
    )

    # Paginate the results
    paginator = Paginator(saved_articles, 10)  # Show 10 saved articles per page
    page_number = request.GET.get("page", 1)
    page_obj = paginator.get_page(page_number)

    context = {
        "page_obj": page_obj,
    }

    return render(request, "news/saved_articles.html", context)


def source_detail_view(request, source_id):
    """
    View function for displaying details about a news source and its articles.
    """
    source = get_object_or_404(Source, id=source_id)

    # Get articles from this source
    articles = Article.objects.filter(source=source).order_by("-publication_date")

    # Paginate the results
    paginator = Paginator(articles, 12)  # Show 12 articles per page
    page_number = request.GET.get("page", 1)
    page_obj = paginator.get_page(page_number)

    context = {
        "source": source,
        "page_obj": page_obj,
    }

    return render(request, "news/source_detail.html", context)


def search_view(request):
    """
    View function for searching articles by keywords.
    """
    query = request.GET.get("q", "")

    if query:
        # Search in title, content, and author
        articles = (
            Article.objects.select_related("source")
            .filter(
                Q(title__icontains=query)
                | Q(content__icontains=query)
                | Q(author__icontains=query)
            )
            .order_by("-publication_date")
        )
    else:
        articles = Article.objects.none()

    # Paginate the results
    paginator = Paginator(articles, 10)  # Show 10 search results per page
    page_number = request.GET.get("page", 1)
    page_obj = paginator.get_page(page_number)

    context = {
        "query": query,
        "page_obj": page_obj,
        "total_results": articles.count(),
    }

    return render(request, "news/search_results.html", context)


def topic_detail_view(request, topic_slug):
    """
    View function for displaying articles related to a specific topic.
    """
    topic = get_object_or_404(Topic, slug=topic_slug)

    # Get articles with this topic
    articles = (
        Article.objects.filter(topics=topic)
        .select_related("source")
        .order_by("-publication_date")
    )

    # Paginate the results
    paginator = Paginator(articles, 12)  # Show 12 articles per page
    page_number = request.GET.get("page", 1)
    page_obj = paginator.get_page(page_number)

    context = {
        "topic": topic,
        "page_obj": page_obj,
    }

    return render(request, "news/topic_detail.html", context)


def detection_model_comparison_view(request):
    """
    View function for displaying performance comparison of different detection models.
    """
    # Get metrics for all models
    model_metrics = DetectionModelMetrics.objects.all().order_by("-accuracy")

    context = {
        "model_metrics": model_metrics,
    }

    return render(request, "news/model_comparison.html", context)
