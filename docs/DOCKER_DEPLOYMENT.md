# VeriFact Docker Deployment Guide

This document provides instructions for deploying the VeriFact application using Docker. This containerized setup includes:
- Django web application
- PostgreSQL database
- Redis for caching and message broker 
- Celery for background task processing
- Nginx for serving the application

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)

## Setup

1. **Clone the repository**

```bash
git clone <repository-url>
cd verifact
```

2. **Configure environment variables**

Copy the example Docker environment file and adjust the settings:

```bash
cp .env.docker .env
```

Edit `.env` to configure your credentials and settings. At minimum, change:
- `SECRET_KEY`: Generate a secure random string
- `DB_PASSWORD`: Set a strong password
- `NEWS_API_KEY`: Your NewsAPI.org API key
- `HUGGINGFACE_API_KEY`: Your Hugging Face API key (if needed)

3. **Create the necessary directories**

```bash
mkdir -p nginx/conf.d logs
```

4. **Build and start the services**

```bash
docker-compose up --build -d
```

This will build and start all the required services in detached mode.

5. **Create a superuser (admin)**

```bash
docker-compose exec web python manage.py createsuperuser
```

6. **Initialize models (optional)**

If you need to initialize the NLP models manually:

```bash
docker-compose exec web python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; AutoTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english'); AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')"
```

7. **Access the application**

The application should now be accessible at http://localhost or the domain name you've configured.

## Management Commands

### View logs

```bash
# Application logs
docker-compose logs -f web

# Celery worker logs
docker-compose logs -f celery_worker

# Celery beat logs
docker-compose logs -f celery_beat
```

### Fetch initial news articles

```bash
docker-compose exec web python manage.py fetch_news --limit 50 --process
```

### Run database migrations

```bash
docker-compose exec web python manage.py migrate
```

### Restart services

```bash
docker-compose restart web
docker-compose restart celery_worker
docker-compose restart celery_beat
```

### Stop all services

```bash
docker-compose down
```

### Stop and remove all data (use with caution)

```bash
docker-compose down -v
```

## Production Deployment Recommendations

For production deployments, consider the following additional steps:

1. **Set up HTTPS**
   - Replace the Nginx configuration with one that includes SSL
   - Obtain an SSL certificate from Let's Encrypt or another provider
   - Update the Docker Compose config to expose port 443

2. **Configure backups**
   - Set up regular PostgreSQL database backups
   - Mount backup volumes to persist data outside of containers

3. **Monitoring**
   - Implement monitoring tools like Prometheus and Grafana
   - Set up alerting for service outages

4. **Scaling**
   - For high-traffic applications, consider scaling the web and worker services
   - Implement a load balancer for multiple web instances

## Troubleshooting

**Database connection issues**
- Check that the database container is running: `docker-compose ps db`
- Verify the database credentials in your `.env` file

**Celery worker not processing tasks**
- Check the Celery worker logs: `docker-compose logs celery_worker`
- Ensure Redis is running: `docker-compose ps redis`

**Static files not loading**
- Make sure you've run `collectstatic`: `docker-compose exec web python manage.py collectstatic --noinput`
- Check the Nginx configuration for static file paths