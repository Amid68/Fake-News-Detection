from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

# Create a router for viewsets
router = DefaultRouter()
router.register(r"articles", views.ArticleViewSet)
router.register(r"sources", views.SourceViewSet)
router.register(r"topics", views.TopicViewSet)
router.register(r"preferences", views.UserPreferenceViewSet, basename="preference")

urlpatterns = [
    # Include router URLs
    path("", include(router.urls)),
    # Additional API endpoints
    path("user/profile/", views.UserProfileView.as_view(), name="api-user-profile"),
    path("analyze/text/", views.AnalyzeTextView.as_view(), name="api-analyze-text"),
    path("models/metrics/", views.ModelMetricsView.as_view(), name="api-model-metrics"),
    path(
        "user/saved-articles/",
        views.SavedArticlesView.as_view(),
        name="api-saved-articles",
    ),
    # Authentication URLs (if using DRF's token auth)
    path("auth/", include("rest_framework.urls")),
]
