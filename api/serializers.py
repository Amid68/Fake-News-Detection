"""
@file api/serializers.py
@brief Serializers for the news aggregator API.

This module provides serializers for converting models to/from JSON
for the REST API.

@author Ameed Othman
@date 2025-04-01
"""

from rest_framework import serializers

from news.models import (
    Article,
    DetectionModelMetrics,
    FakeNewsDetectionResult,
    Source,
    Topic,
)
from users.models import CustomUser, UserPreference


class SourceSerializer(serializers.ModelSerializer):
    """Serializer for news sources."""

    class Meta:
        model = Source
        fields = [
            "id",
            "name",
            "base_url",
            "description",
            "logo_url",
            "reliability_score",
        ]


class TopicSerializer(serializers.ModelSerializer):
    """Serializer for topics."""

    class Meta:
        model = Topic
        fields = ["id", "name", "slug", "description"]


class FakeNewsDetectionResultSerializer(serializers.ModelSerializer):
    """Serializer for fake news detection results."""

    class Meta:
        model = FakeNewsDetectionResult
        fields = [
            "credibility_score",
            "credibility_category",
            "confidence",
            "model_name",
            "processing_time",
            "explanation",
        ]


class ArticleSerializer(serializers.ModelSerializer):
    """Serializer for article list view."""

    source = SourceSerializer(read_only=True)
    topics = TopicSerializer(many=True, read_only=True)
    has_detection = serializers.SerializerMethodField()

    class Meta:
        model = Article
        fields = [
            "id",
            "title",
            "summary",
            "source",
            "author",
            "publication_date",
            "source_article_url",
            "featured_image_url",
            "bias_score",
            "topics",
            "view_count",
            "has_detection",
        ]

    def get_has_detection(self, obj):
        """Check if article has fake news detection results."""
        return hasattr(obj, "fake_news_detection")


class ArticleDetailSerializer(ArticleSerializer):
    """Serializer for article detail view."""

    detection_result = serializers.SerializerMethodField()
    content = serializers.CharField()

    class Meta(ArticleSerializer.Meta):
        fields = ArticleSerializer.Meta.fields + ["content", "detection_result"]

    def get_detection_result(self, obj):
        """Get fake news detection results if available."""
        if hasattr(obj, "fake_news_detection"):
            return FakeNewsDetectionResultSerializer(obj.fake_news_detection).data
        return None


class UserPreferenceSerializer(serializers.ModelSerializer):
    """Serializer for user preferences."""

    class Meta:
        model = UserPreference
        fields = ["id", "topic_keyword"]
        read_only_fields = ["id"]

    def validate_topic_keyword(self, value):
        """Validate that the topic is not already preferred by the user."""
        user = self.context["request"].user

        # Convert to lowercase for consistency
        value = value.lower()

        # Check existing preferences
        if UserPreference.objects.filter(
            user=user, topic_keyword__iexact=value
        ).exists():
            raise serializers.ValidationError("You've already added this topic.")

        return value


class UserSerializer(serializers.ModelSerializer):
    """Serializer for user profiles."""

    preferences = UserPreferenceSerializer(many=True, read_only=True)

    class Meta:
        model = CustomUser
        fields = ["id", "username", "email", "first_name", "last_name", "preferences"]
        read_only_fields = ["id", "username", "preferences"]


class DetectionModelMetricsSerializer(serializers.ModelSerializer):
    """Serializer for model performance metrics."""

    class Meta:
        model = DetectionModelMetrics
        fields = [
            "model_name",
            "accuracy",
            "precision_score",
            "recall_score",
            "f1_score",
            "avg_processing_time",
            "avg_memory_usage",
            "parameter_count",
            "efficiency_score",
            "updated_at",
        ]


class AnalyzeTextSerializer(serializers.Serializer):
    """Serializer for text analysis requests."""

    text = serializers.CharField(required=True, max_length=10000)
    model = serializers.CharField(required=False, default="distilbert")

    def validate_model(self, value):
        """Validate that the model key is supported."""
        valid_models = ["distilbert", "tinybert", "mobilebert", "albert"]

        if value not in valid_models:
            raise serializers.ValidationError(
                f"Unsupported model. Choose from: {', '.join(valid_models)}"
            )

        return value
