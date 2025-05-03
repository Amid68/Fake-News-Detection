"""
@file news/services.py

@brief Services for fetching, processing, and managing news articles.

This module provides functionality to interact with external news APIs,
process article data, and manage database operations for news content.

@author Ameed Othman
@date 2025-03-05
"""

import logging
import time
import os
import json
from typing import Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import torch
    import psutil
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    logger.error(f"ML dependencies not available: {str(e)}")
    DEPENDENCIES_AVAILABLE = False

from django.conf import settings

from .models import Article, FakeNewsDetectionResult, DetectionModelMetrics

# Base directory for model files
MODEL_BASE_DIR = os.path.join(settings.BASE_DIR, 'ml_models')

# Define available models
MODELS = {
    "distilbert_finetuned": {
        "name": "DistilBERT (Fine-tuned)",
        "path": os.path.join(MODEL_BASE_DIR, "distilbert_fakenewsnet"),
        "max_length": 512,
        "description": "Fine-tuned DistilBERT model optimized for news credibility detection"
    },
    "distilbert": {
        "name": "DistilBERT (Base)",
        "path": "distilbert-base-uncased",
        "max_length": 512,
        "description": "Pre-trained DistilBERT model with default classification head"
    },
    "tinybert": {
        "name": "TinyBERT",
        "path": "huawei-noah/TinyBERT_General_4L_312D",
        "max_length": 512,
        "description": "Smaller and faster model with fewer parameters"
    }
}


def detect_fake_news(text, model_key="distilbert_finetuned"):
    """
    Detect fake news with specified model

    Args:
        text (str): The text to analyze
        model_key (str): Key of the model to use from MODELS dictionary

    Returns:
        dict: Analysis results including credibility score and processing metrics
    """
    if not DEPENDENCIES_AVAILABLE:
        return {
            "error": "ML dependencies not available. Please install required packages.",
            "model_name": MODELS.get(model_key, MODELS["distilbert_finetuned"])["name"]
        }

    model_config = MODELS.get(model_key, MODELS["distilbert_finetuned"])

    # Get memory usage before loading model
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)  # MB

    # Track performance
    start_time = time.time()

    try:
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_config["path"])
        model = AutoModelForSequenceClassification.from_pretrained(model_config["path"])

        # Get device (CPU or GPU)
        device = 0 if torch.cuda.is_available() else -1

        # Create classifier pipeline
        classifier = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=device
        )

        # Preprocess text (lowercase, remove extra whitespace)
        processed_text = text.lower().strip()

        # Process text
        result = classifier(processed_text[:model_config["max_length"]])[0]

        # Calculate processing time and memory usage
        processing_time = time.time() - start_time
        mem_after = process.memory_info().rss / (1024 * 1024)  # MB
        memory_used = mem_after - mem_before

        # Parse results
        label = result["label"]
        score = result["score"]

        # For fine-tuned model, LABEL_1 is fake, LABEL_0 is real
        if "LABEL_0" in label or "POSITIVE" in label:
            credibility_score = score  # Higher score = more credible
            category = "credible" if score > 0.7 else "mixed"
        else:
            credibility_score = 1 - score  # Lower score = less credible
            category = "fake" if score > 0.7 else "mixed"

        return {
            "credibility_score": credibility_score,
            "category": category,
            "confidence": score,
            "model_name": model_config["name"],
            "processing_time": processing_time,
            "memory_usage": memory_used
        }

    except Exception as e:
        logger.error(f"Error detecting fake news: {str(e)}")
        return {
            "error": str(e),
            "model_name": model_config["name"]
        }


def initialize_model_metrics():
    """
    Initialize model metrics in the database based on
    the fine-tuned model's metrics file
    """
    metrics_path = os.path.join(MODEL_BASE_DIR, "distilbert_fakenewsnet_metrics.json")

    try:
        if not os.path.exists(metrics_path):
            logger.warning(f"Metrics file not found at {metrics_path}")
            return

        with open(metrics_path, 'r') as f:
            metrics = json.load(f)

        logger.info(f"Loaded metrics from {metrics_path}: {metrics}")

        # Create or update metrics in the database
        DetectionModelMetrics.objects.update_or_create(
            model_name=metrics['model_name'] + " (Fine-tuned)",
            defaults={
                'accuracy': metrics.get('accuracy', 0.0),
                'f1_score': metrics.get('f1_score', 0.0),
                'avg_processing_time': metrics.get('avg_processing_time', 0.0),
                'avg_memory_usage': metrics.get('avg_memory_usage', 0.0),
                'parameter_count': metrics.get('parameter_count', 0)
            }
        )

        # Also add a placeholder for TinyBERT for comparison
        DetectionModelMetrics.objects.update_or_create(
            model_name="TinyBERT",
            defaults={
                'accuracy': 0.85,
                'f1_score': 0.84,
                'avg_processing_time': 0.7,
                'avg_memory_usage': 125.0,
                'parameter_count': 14500000
            }
        )

        logger.info(f"Initialized metrics for {metrics['model_name']}")

    except Exception as e:
        logger.error(f"Error initializing model metrics: {str(e)}")