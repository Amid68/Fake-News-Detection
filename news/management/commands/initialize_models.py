# news/management/commands/initialize_models.py

import os
from django.core.management.base import BaseCommand
from django.conf import settings
from news.services import initialize_model_metrics


class Command(BaseCommand):
    help = 'Initialize ML models and metrics'

    def add_arguments(self, parser):
        parser.add_argument(
            '--recreate',
            action='store_true',
            help='Recreate metrics even if they exist',
        )

    def handle(self, *args, **options):
        # Create models directory if it doesn't exist
        model_dir = os.path.join(settings.BASE_DIR, 'ml_models')
        os.makedirs(model_dir, exist_ok=True)

        # Check LIAR2 model path
        liar2_model_path = os.path.join(model_dir, 'distilbert_LIAR2')
        if os.path.exists(liar2_model_path):
            self.stdout.write(self.style.SUCCESS(f'LIAR2 model found at {liar2_model_path}'))
        else:
            self.stdout.write(self.style.WARNING(f'LIAR2 model NOT found at {liar2_model_path}'))

        # Initialize metrics from services
        self.stdout.write(self.style.SUCCESS('Initializing model metrics...'))
        initialize_model_metrics()

        self.stdout.write(self.style.SUCCESS('Successfully initialized model metrics'))