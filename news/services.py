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
from typing import Dict, Any

from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from .models import Article, FakeNewsDetectionResult, DetectionModelMetrics

"""Services for fake news detection model comparison"""
import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

MODELS = {
    "distilbert": {
        "name": "DistilBERT",
        "path": "distilbert-base-uncased",
        "max_length": 512
    },
    "tinybert": {
        "name": "TinyBERT",
        "path": "huawei-noah/TinyBERT_General_4L_312D",
        "max_length": 512
    }
}


def detect_fake_news(text, model_key="distilbert"):
    """Detect fake news with specified model"""
    model_config = MODELS[model_key]

    # Track performance
    start_time = time.time()

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_config["path"])
    model = AutoModelForSequenceClassification.from_pretrained(model_config["path"], num_labels=2)

    # Create classifier
    classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    # Process text
    result = classifier(text[:model_config["max_length"]])[0]

    # Calculate processing time
    processing_time = time.time() - start_time

    # Map results to credibility score
    label = result["label"]
    score = result["score"]

    if "POSITIVE" in label:
        credibility_score = score
        category = "credible" if score > 0.7 else "mixed"
    else:
        credibility_score = 1 - score
        category = "fake" if score > 0.7 else "mixed"

    return {
        "credibility_score": credibility_score,
        "category": category,
        "confidence": score,
        "model_name": model_config["name"],
        "processing_time": processing_time
    }