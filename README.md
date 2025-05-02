# LightFakeDetect: Lightweight Fake News Detection System

A Django-based web application that demonstrates fake news detection using lightweight pre-trained models.

## Core Features

- **Fake News Detection:** Analyze text using multiple lightweight ML models
- **Model Comparison:** Compare performance metrics between different models
- **Interactive Demo:** Test any text with different detection models

## Technology Stack

### Backend
- Python 3.10+
- Django 5.1+
- SQLite for database

### Natural Language Processing
- Hugging Face Transformers
- Two comparison models: DistilBERT and TinyBERT
- Optimized for low resource usage

### Frontend
- Bootstrap 5
- Chart.js for data visualization
- Basic HTML/CSS/JavaScript

## Model Comparison

LightFakeDetect demonstrates lightweight models for fake news detection, optimized for different use cases:

| Model | Accuracy | F1 Score | Memory Usage | Processing Time |
|-------|----------|----------|--------------|-----------------|
| DistilBERT | 0.99 | 0.99 | 330 MB | 1.2s |
| TinyBERT | 0.85 | 0.84 | 125 MB | 0.7s |

## Quick Start

### Prerequisites

- Python 3.10 or higher
- pip and virtualenv

### Installation

```bash
# Clone the repository
git clone https://github.com/Amid68/LightFakeDetect.git
cd LightFakeDetect

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

## Project Structure

```
LightFakeDetect/
├── news/                     # News content and detection
│   ├── models.py             # Database models
│   ├── services.py           # Detection services
│   └── views.py              # View functions
├── templates/                # HTML templates
│   ├── news/
│   │   ├── model_comparison.html
│   │   └── analyze_text.html
└── static/                   # Static assets
```

## Usage

### Key Pages

- **Model Comparison:** Compare different detection model performance
- **Analyze Text:** Test any text with the detection models

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for NLP models and tools
- [Django](https://www.djangoproject.com/) for the web framework
- [Chart.js](https://www.chartjs.org/) for visualizations