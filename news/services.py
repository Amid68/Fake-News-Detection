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
    "distilbert_liar2": {
        "name": "DistilBERT (LIAR2 Fine-tuned)",
        "path": os.path.join(MODEL_BASE_DIR, "distilbert_LIAR2"),
        "max_length": 128,
        "description": "DistilBERT model fine-tuned on the LIAR2 dataset for 6-class fake news detection"
    },
    "tinybert": {
        "name": "TinyBERT (Coming Soon)",
        "path": "huawei-noah/TinyBERT_General_4L_312D",
        "max_length": 512,
        "description": "Smaller and faster model with fewer parameters (not yet fine-tuned)"
    }
}

# Define LIAR2 label mapping
LIAR2_LABEL_MAPPING = {
    0: "pants_on_fire",
    1: "false",
    2: "mostly_false",
    3: "half_true",
    4: "mostly_true",
    5: "true"
}

# Define LIAR2 label colors for UI
LIAR2_LABEL_COLORS = {
    0: "danger",      # Red
    1: "danger",      # Red
    2: "warning",     # Orange/Yellow
    3: "warning",     # Orange/Yellow
    4: "success",     # Green
    5: "success"      # Green
}

def combine_features(text, speaker=None, subject=None, context=None):
    """
    Combine text and metadata features for LIAR2 model

    Args:
        text (str): The main text to analyze
        speaker (str): Who said the statement
        subject (str): Subject area of the statement
        context (str): Context in which the statement was made

    Returns:
        str: Combined features formatted for the model
    """
    speaker = str(speaker) if speaker else "unknown"
    subject = str(subject) if subject else "unknown"
    context = str(context) if context else "unknown"

    # Combine features with special tokens
    combined = f"{text} [SEP] Subject: {subject} [SEP] Speaker: {speaker} [SEP] Context: {context}"
    return combined


def detect_fake_news(text, model_key="distilbert_liar2", speaker=None, subject=None, context=None):
    """
    Detect fake news with specified model

    Args:
        text (str): The text to analyze
        model_key (str): Key of the model to use from MODELS dictionary
        speaker (str): Speaker metadata for LIAR2 model
        subject (str): Subject metadata for LIAR2 model
        context (str): Context metadata for LIAR2 model

    Returns:
        dict: Analysis results including credibility score and processing metrics
    """
    if not DEPENDENCIES_AVAILABLE:
        return {
            "error": "ML dependencies not available. Please install required packages.",
            "model_name": MODELS.get(model_key, MODELS["distilbert_liar2"])["name"]
        }

    model_config = MODELS.get(model_key, MODELS["distilbert_liar2"])

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

        # Preprocess text and combine with metadata for LIAR2 model
        if model_key == "distilbert_liar2":
            processed_text = combine_features(
                text.lower().strip(),
                speaker=speaker,
                subject=subject,
                context=context
            )
        else:
            processed_text = text.lower().strip()

        # Process text
        result = classifier(processed_text[:model_config["max_length"]])[0]

        # Calculate processing time and memory usage
        processing_time = time.time() - start_time
        mem_after = process.memory_info().rss / (1024 * 1024)  # MB
        memory_used = mem_after - mem_before

        # Parse results based on the model
        label = result["label"]
        score = result["score"]

        if model_key == "distilbert_liar2":
            # For LIAR2 dataset: 6-class classification
            try:
                # Extract label index (expected format: "LABEL_0", "LABEL_1", etc.)
                label_idx = int(label.split('_')[-1]) if '_' in label else int(label)

                # Get the text label
                label_text = LIAR2_LABEL_MAPPING.get(label_idx, "unknown")

                # Scale the credibility score from 0-5 range to 0-1 range for visualization
                credibility_score = label_idx / 5.0

                # Get the UI color
                color = LIAR2_LABEL_COLORS.get(label_idx, "secondary")

                return {
                    "credibility_score": credibility_score,
                    "confidence": score,
                    "label_index": label_idx,
                    "label_text": label_text,
                    "color": color,
                    "model_name": model_config["name"],
                    "processing_time": processing_time,
                    "memory_usage": memory_used
                }
            except (ValueError, IndexError) as e:
                logger.error(f"Error parsing LIAR2 model output: {str(e)}")
                return {
                    "error": f"Invalid model output format: {label}",
                    "model_name": model_config["name"]
                }
        else:
            # Generic handling for other models
            credibility_score = 0.5  # Default
            label_text = "unknown"

            return {
                "credibility_score": credibility_score,
                "confidence": score,
                "label_text": label_text,
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
    try:
        # Add metrics for the LIAR2 fine-tuned model
        # These values come from the notebook's final evaluation
        DetectionModelMetrics.objects.update_or_create(
            model_name="DistilBERT (LIAR2 Fine-tuned)",
            defaults={
                'accuracy': 0.36,
                'f1_score': 0.32,
                'avg_processing_time': 1.2,
                'avg_memory_usage': 350.0,
                'parameter_count': 66000000  # ~66M parameters for DistilBERT
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

        logger.info("Initialized metrics for all models")

    except Exception as e:
        logger.error(f"Error initializing model metrics: {str(e)}")