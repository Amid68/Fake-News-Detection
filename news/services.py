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
import psutil
from typing import Dict, Any
from pathlib import Path

from django.conf import settings
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

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
        text = text.lower().strip()

        # Process text
        result = classifier(text[:model_config["max_length"]])[0]

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
        logging.error(f"Error detecting fake news: {str(e)}")
        return {
            "error": str(e),
            "model_name": model_config["name"]
        }


def initialize_model_metrics():
    """
    Initialize model metrics in the database based on
    the fine-tuned model's metrics file
    """
    import json

    metrics_path = os.path.join(MODEL_BASE_DIR, "distilbert_fakenewsnet_metrics.json")

    try:
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)

        # Create or update metrics in the database
        DetectionModelMetrics.objects.update_or_create(
            model_name=metrics['model_name'],
            defaults={
                'accuracy': metrics.get('accuracy', 0.0),
                'f1_score': metrics.get('f1_score', 0.0),
                'avg_processing_time': metrics.get('avg_processing_time', 0.0),
                'avg_memory_usage': metrics.get('avg_memory_usage', 0.0),
                'parameter_count': metrics.get('parameter_count', 0)
            }
        )
        logging.info(f"Initialized metrics for {metrics['model_name']}")

    except Exception as e:
        logging.error(f"Error initializing model metrics: {str(e)}")