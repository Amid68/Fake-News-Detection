# news/management/commands/initialize_models.py

import os
from django.core.management.base import BaseCommand
from django.conf import settings
from news.services import initialize_model_metrics


class Command(BaseCommand):
    help = 'Initialize ML models and metrics'

    def handle(self, *args, **options):
        # Create models directory if it doesn't exist
        model_dir = os.path.join(settings.BASE_DIR, 'ml_models')
        os.makedirs(model_dir, exist_ok=True)

        # Initialize metrics from JSON file
        self.stdout.write(self.style.SUCCESS('Initializing model metrics...'))
        initialize_model_metrics()

        self.stdout.write(self.style.SUCCESS('Successfully initialized model metrics'))