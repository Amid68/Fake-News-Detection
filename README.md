# VeriFact: Lightweight Fake News Detection System

VeriFact is a simplified Django-based web application that demonstrates fake news detection using lightweight pre-trained models, ideal for academic demonstration and research presentation.

## Core Features

- **User Authentication:** Basic registration and login system
- **News Display:** View curated news articles with detailed view
- **Fake News Detection:** Analyze text using lightweight ML models
- **Model Comparison:** Compare performance metrics between models
- **Interactive Demo:** Test any text with different detection models

## Simplified Technology Stack

### Backend
- Python 3.10+
- Django 5.1+
- SQLite for database

### Natural Language Processing
- Hugging Face Transformers
- Two comparison models: DistilBERT and TinyBERT
- Optimized for demonstration purposes

### Frontend
- Bootstrap 5
- Chart.js for data visualization
- Basic HTML/CSS/JavaScript

## Model Comparison

VeriFact demonstrates lightweight models for fake news detection, optimized for different use cases:

| Model | Accuracy | F1 Score | Memory Usage | Processing Time |
|-------|----------|----------|--------------|-----------------|
| DistilBERT | 0.89 | 0.88 | 330 MB | 1.2s |
| TinyBERT | 0.85 | 0.84 | 125 MB | 0.7s |

## Quick Start

### Prerequisites

- Python 3.10 or higher
- pip and virtualenv

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/verifact.git
cd verifact

# Set up virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Initialize database
python manage.py migrate
python manage.py createsuperuser

# Start the development server
python manage.py runserver
```

## Simplified Project Structure

```
verifact/
├── news/                     # News content and detection
│   ├── migrations/           # Database migrations
│   ├── models.py             # Database models (simplified)
│   ├── services.py           # Detection services
│   └── views.py              # View functions
├── news_aggregator/          # Project settings
├── templates/                # HTML templates
│   ├── news/
│   │   ├── home.html
│   │   ├── article_detail.html
│   │   ├── model_comparison.html
│   │   └── analyze_text.html
└── users/                    # Basic user authentication
```

## Usage

### Access Key Pages

- **Home:** View the list of news articles
- **Article Detail:** View article content and any detection results
- **Model Comparison:** Compare different detection model performance
- **Analyze Text:** Test any text with the detection models

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for NLP models and tools
- [Django](https://www.djangoproject.com/) for the web framework
- [Chart.js](https://www.chartjs.org/) for visualizations