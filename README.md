# Automated Multilingual News Aggregator

A Django-based web application for aggregating, summarizing, and detecting bias in news articles.

## Features

- **User Authentication:** Secure registration and login system with email verification
- **News Aggregation:** Collects articles from reliable English language sources
- **Content Analysis:** AI-powered article summarization and bias detection
- **Personalized Feed:** News feed based on user topic preferences
- **Responsive Design:** Clean, intuitive interface that works on desktop and mobile

## Technology Stack

### Backend
- Python 3.10+
- Django 5.1+
- Django REST Framework
- PostgreSQL (production) / SQLite (development)
- Celery for asynchronous tasks
- Redis for caching and task queue

### Natural Language Processing
- Hugging Face Transformers
- BART/T5 for summarization
- BERT/RoBERTa for bias detection

### Frontend
- HTML/CSS/JavaScript
- Future: React-based SPA (in development)

## Prerequisites

- Python 3.10 or higher
- pip and virtualenv
- Redis server (for Celery and caching)
- PostgreSQL (for production)
- NewsAPI key

## Installation

### 1. Clone the Repository

```bash
git clone git@github.com:Amid68/news_bias_detection.git
cd news_bias_detection
```

### 2. Set Up Environment

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Copy the example environment file and update it with your settings:

```bash
cp .env.example .env
```

Open `.env` in your editor and set the necessary variables:

```
# Essential settings to update:
SECRET_KEY=your-secret-key-here
DEBUG=True  # Set to False in production
NEWS_API_KEY=your-newsapi-key-here
```

### 4. Initialize the Database

Run migrations:

```bash
python manage.py migrate
```

Create a superuser:

```bash
python manage.py createsuperuser
```

### 5. Start the Development Server

```bash
python manage.py runserver
```

The application should now be running at http://127.0.0.1:8000/

### 6. Set Up Celery Worker (Optional for Development)

In a new terminal window with the virtual environment activated:

```bash
celery -A news_aggregator worker -l info
```

For the task scheduler (beat):

```bash
celery -A news_aggregator beat -l info
```

## Usage

### Fetching News Articles

```bash
# Fetch latest news
python manage.py fetch_news

# Fetch with specific parameters
python manage.py fetch_news --category technology --limit 50 --process
```

### Processing Articles

```bash
# Queue summarization for all unprocessed articles
python manage.py process_articles --type summarization

# Queue bias detection for all unprocessed articles
python manage.py process_articles --type bias_detection
```

## Project Structure

```
news_aggregator/              # Main project folder
├── api/                      # REST API app
├── docs/                     # Project documentation
├── news/                     # News content management app
│   ├── management/           # Custom management commands
│   ├── migrations/           # Database migrations
│   ├── models.py             # Database models
│   ├── services.py           # Business logic and services
│   └── views.py              # View functions
├── processing/               # NLP processing app
│   ├── models.py             # Processing models
│   ├── services.py           # Processing services
│   └── tasks.py              # Celery tasks
├── news_aggregator/          # Project settings
├── static/                   # Static files (CSS, JS)
├── templates/                # HTML templates
└── users/                    # User management app
```

## Development

### Running Tests

```bash
python manage.py test
```

### Code Quality

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run linting
flake8

# Run type checking
mypy .
```

## Deployment

### Docker (Recommended)

1. Build the Docker image:
   ```bash
   docker-compose build
   ```

2. Start the services:
   ```bash
   docker-compose up -d
   ```

3. Run migrations:
   ```bash
   docker-compose exec web python manage.py migrate
   ```

### Traditional Deployment

For production deployment, consider:

1. Using Gunicorn as the WSGI server
2. Setting up Nginx as a reverse proxy
3. Configuring PostgreSQL for the database
4. Setting up Redis for caching and Celery
5. Using supervisor for process management

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [NewsAPI](https://newsapi.org/) for providing the news data
- [Hugging Face](https://huggingface.co/) for NLP models and tools
- [Django](https://www.djangoproject.com/) for the web framework