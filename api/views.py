"""
@file api/views.py
@brief API views for the news aggregator application.

This module provides REST API endpoints for accessing news articles,
user preferences, and fake news detection functionality.

@author Ameed Othman
@date 2025-04-01
"""

from rest_framework import viewsets, status, filters
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated, IsAuthenticatedOrReadOnly
from rest_framework.decorators import action
from django.shortcuts import get_object_or_404
from django.db.models import Q

from news.models import Article, Source, Topic, FakeNewsDetectionResult, DetectionModelMetrics
from users.models import UserPreference, CustomUser
from processing.services import detect_fake_news
from .serializers import (
    ArticleSerializer, ArticleDetailSerializer, SourceSerializer,
    TopicSerializer, UserPreferenceSerializer, FakeNewsDetectionResultSerializer,
    DetectionModelMetricsSerializer, UserSerializer, AnalyzeTextSerializer
)


class ArticleViewSet(viewsets.ReadOnlyModelViewSet):
    """API endpoint for viewing news articles."""
    queryset = Article.objects.all().order_by('-publication_date')
    permission_classes = [IsAuthenticatedOrReadOnly]
    filter_backends = [filters.SearchFilter]
    search_fields = ['title', 'content']

    def get_serializer_class(self):
        if self.action == 'retrieve':
            return ArticleDetailSerializer
        return ArticleSerializer

    def get_queryset(self):
        queryset = super().get_queryset()

        # Filter by source if provided
        source_id = self.request.query_params.get('source')
        if source_id:
            queryset = queryset.filter(source_id=source_id)

        # Filter by topic if provided
        topic_slug = self.request.query_params.get('topic')
        if topic_slug:
            queryset = queryset.filter(topics__slug=topic_slug)

        # Personalized feed if user is authenticated and no other filters
        if self.request.user.is_authenticated and not (source_id or topic_slug):
            if hasattr(self.request.query_params, 'personalized') and self.request.query_params.get(
                    'personalized') == 'true':
                # Get user preferences
                preferences = UserPreference.objects.filter(user=self.request.user)
                if preferences.exists():
                    # Build topic query
                    topic_query = Q()
                    for pref in preferences:
                        topic_query |= Q(content__icontains=pref.topic_keyword) | Q(title__icontains=pref.topic_keyword)

                    # Apply filter if we have preferences
                    if topic_query:
                        queryset = queryset.filter(topic_query)

        return queryset

    @action(detail=True, methods=['post'], permission_classes=[IsAuthenticated])
    def save(self, request, pk=None):
        """Save an article for later reading."""
        article = self.get_object()

        from news.models import UserSavedArticle
        _, created = UserSavedArticle.objects.get_or_create(
            user=request.user,
            article=article
        )

        if created:
            return Response({'status': 'article saved'})
        else:
            return Response({'status': 'article already saved'})

    @action(detail=True, methods=['post'], permission_classes=[IsAuthenticated])
    def unsave(self, request, pk=None):
        """Remove an article from saved list."""
        article = self.get_object()

        from news.models import UserSavedArticle
        deleted, _ = UserSavedArticle.objects.filter(
            user=request.user,
            article=article
        ).delete()

        if deleted:
            return Response({'status': 'article removed from saved list'})
        else:
            return Response({'status': 'article was not in saved list'})


class SourceViewSet(viewsets.ReadOnlyModelViewSet):
    """API endpoint for viewing news sources."""
    queryset = Source.objects.all().order_by('name')
    serializer_class = SourceSerializer
    permission_classes = [IsAuthenticatedOrReadOnly]


class TopicViewSet(viewsets.ReadOnlyModelViewSet):
    """API endpoint for viewing topics."""
    queryset = Topic.objects.all().order_by('name')
    serializer_class = TopicSerializer
    permission_classes = [IsAuthenticatedOrReadOnly]
    lookup_field = 'slug'


class UserPreferenceViewSet(viewsets.ModelViewSet):
    """API endpoint for managing user preferences."""
    serializer_class = UserPreferenceSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return UserPreference.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)


class UserProfileView(APIView):
    """API endpoint for managing user profile."""
    permission_classes = [IsAuthenticated]

    def get(self, request):
        serializer = UserSerializer(request.user)
        return Response(serializer.data)

    def put(self, request):
        serializer = UserSerializer(request.user, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class AnalyzeTextView(APIView):
    """API endpoint to analyze text for fake news."""
    permission_classes = [IsAuthenticated]

    def post(self, request):
        serializer = AnalyzeTextSerializer(data=request.data)
        if serializer.is_valid():
            text = serializer.validated_data['text']
            model_key = serializer.validated_data.get('model', 'distilbert')

            # Analyze the text
            try:
                results = detect_fake_news(text, model_key)
                return Response(results)
            except Exception as e:
                return Response(
                    {'error': f'Analysis failed: {str(e)}'},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class ModelMetricsView(APIView):
    """API endpoint to get detection model metrics."""
    permission_classes = [IsAuthenticatedOrReadOnly]

    def get(self, request):
        metrics = DetectionModelMetrics.objects.all().order_by('-f1_score')
        serializer = DetectionModelMetricsSerializer(metrics, many=True)
        return Response(serializer.data)


class SavedArticlesView(APIView):
    """API endpoint to get user's saved articles."""
    permission_classes = [IsAuthenticated]

    def get(self, request):
        from news.models import UserSavedArticle
        saved = UserSavedArticle.objects.filter(
            user=request.user
        ).select_related('article').order_by('-saved_at')

        articles = [item.article for item in saved]
        serializer = ArticleSerializer(articles, many=True)
        return Response(serializer.data)