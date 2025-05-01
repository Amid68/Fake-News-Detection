# VeriFact: Lightweight Fake News Detection System

VeriFact is a Django-based web application for aggregating, summarizing, and detecting bias in news articles, with a focus on efficient fake news detection using lightweight pre-trained models.

## Features

- **User Authentication:** Secure registration and login system with preference management
- **News Aggregation:** Collects articles from reliable English language sources
- **Fake News Detection:** Analyzes articles using lightweight pre-trained models
- **Model Comparison:** Compare performance and resource usage of different detection models
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
- DistilBERT, TinyBERT, MobileBERT, and ALBERT models
- Optimized for resource efficiency

### Frontend
- Bootstrap 5
- Chart.js for data visualization
- HTML/CSS/JavaScript

## Screenshots

![Home Screen](docs/images/home-screen.png)
![Article Detail](docs/images/article-detail.png)
![Model Comparison](docs/images/model-comparison.png)

## Quick Start

### Prerequisites

- Python 3.10 or higher
- pip and virtualenv
- Redis server (for Celery and caching)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/verifact.git
cd verifact

# Set up virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Edit .env with your settings

# Initialize database
python manage.py migrate
python manage.py createsuperuser

# Start the development server
python manage.py runserver
```

For detailed setup instructions, see [SETUP.md](SETUP.md).

## Model Comparison

VeriFact includes several lightweight models for fake news detection, optimized for different use cases:

| Model | Accuracy | F1 Score | Memory Usage | Processing Time |
|-------|----------|----------|--------------|-----------------|
| DistilBERT | 0.89 | 0.88 | 330 MB | 1.2s |
| TinyBERT | 0.85 | 0.84 | 125 MB | 0.7s |
| MobileBERT | 0.87 | 0.86 | 190 MB | 0.9s |
| ALBERT | 0.83 | 0.82 | 70 MB | 0.5s |

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
verifact/
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
├── templates/                # HTML templates
└── users/                    # User management app
```

## Testing

Run the test suite:

```bash
python manage.py test
```

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