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
            # Make sure custom template tags are available
            from django.template.base import add_to_builtins
            add_to_builtins('news.templatetags.custom_filters')

            # Initialize model metrics
            from .services import initialize_model_metrics
            initialize_model_metrics()