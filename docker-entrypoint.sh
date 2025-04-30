#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Apply database migrations
echo "Applying database migrations..."
python manage.py migrate --noinput

# Collect static files
echo "Collecting static files..."
python manage.py collectstatic --noinput

# Create default admin user if provided in environment
if [ -n "$DJANGO_SUPERUSER_USERNAME" ] && [ -n "$DJANGO_SUPERUSER_PASSWORD" ] && [ -n "$DJANGO_SUPERUSER_EMAIL" ]; then
    echo "Creating/updating superuser..."
    python manage.py createsuperuser --noinput
fi

# Load initial data if needed
if [ "$LOAD_INITIAL_DATA" = "True" ]; then
    echo "Loading initial data..."
    python manage.py loaddata initial_data
fi

# Check if we need to download NLP models
if [ "$DOWNLOAD_NLP_MODELS" = "True" ]; then
    echo "Downloading NLP models..."
    python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; AutoTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english'); AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')"
fi

exec "$@"