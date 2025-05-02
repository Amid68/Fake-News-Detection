# news/apps.py
from django.apps import AppConfig


class NewsConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "news"

    def ready(self):
        """
        Initialize model metrics when the app is ready
        """
        # Avoid importing during Django's app registry loading
        import os
        if os.environ.get('RUN_MAIN', None) == 'true':
            from .services import initialize_model_metrics
            initialize_model_metrics()