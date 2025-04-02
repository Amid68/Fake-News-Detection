# VeriFact Setup Guide

This document provides step-by-step instructions for setting up the VeriFact application, a news aggregator with fake news detection capabilities.

## Prerequisites

Ensure you have the following installed:

- Python 3.10 or higher
- pip and virtualenv
- PostgreSQL (for production) or SQLite (for development)
- Redis server (for Celery and caching)
- Git

## Installation Steps

### 1. Clone the Repository

```bash
git clone <repository-url>
cd verifact
```

### 2. Set Up Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Copy the example environment file and update it with your settings:

```bash
cp .env.example .env
```

Open the `.env` file in your text editor and update the following required settings:

- `SECRET_KEY`: Generate a secure random key
- `DEBUG`: Set to `True` for development, `False` for production
- `NEWS_API_KEY`: Your API key from NewsAPI
- `HUGGINGFACE_API_KEY`: Your API key from Hugging Face (for accessing models)

For production environments, also configure:
- Database settings
- Email settings
- Redis settings

### 5. Initialize the Database

Run migrations to set up the database schema:

```bash
python manage.py migrate
```

Create a superuser to access the admin interface:

```bash
python manage.py createsuperuser
```

### 6. Load Initial Data (Optional)

If you want to start with some test data:

```bash
python manage.py loaddata initial_data
```

### 7. Set Up Celery Worker (Required for Processing)

Start the Celery worker for async tasks:

```bash
celery -A news_aggregator worker -l info
```

For the task scheduler (optional):

```bash
celery -A news_aggregator beat -l info
```

### 8. Start the Development Server

```bash
python manage.py runserver
```

The application should now be running at http://127.0.0.1:8000/

## First-Time Setup

After installing, follow these steps to set up your VeriFact instance:

1. **Configure News Sources**:
   - Log in to the admin interface at `/admin/`
   - Add news sources under the News section

2. **Fetch Initial News Articles**:
   ```bash
   python manage.py fetch_news --limit 50 --process
   ```

3. **Create Topics** (in the admin interface):
   - Add common topics such as "Politics", "Technology", "Business", etc.

## Production Deployment

For production deployment, consider the following additional steps:

1. Use a proper WSGI server like Gunicorn:
   ```bash
   pip install gunicorn
   gunicorn news_aggregator.wsgi:application
   ```

2. Set up Nginx as a reverse proxy

3. Configure SSL certificates

4. Set up proper logging and monitoring

5. Configure database backups

## Common Issues and Troubleshooting

### Model Loading Errors

If you encounter errors when loading models:

1. Check your Hugging Face API key
2. Ensure you have enough disk space for the model files
3. Try using smaller models (e.g., TinyBERT instead of DistilBERT)

### News API Rate Limits

If you hit rate limits with the News API:

1. Reduce the frequency of article fetching
2. Implement proper caching strategies
3. Consider upgrading your API subscription

## Additional Configuration

### Customizing Detection Models

You can customize which models are used by editing the `MODELS` dictionary in `processing/services.py`.

### Scheduling Regular Updates

To set up scheduled tasks for fetching news and processing:

1. Install and configure Celery Beat
2. Create periodic tasks in the admin interface