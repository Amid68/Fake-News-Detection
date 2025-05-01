"""
@file processing/services.py
@brief Services for processing news articles with NLP models.

This module provides functionality to analyze news articles for bias
and generate summaries using lightweight NLP models.

@author Ameed Othman
@date 2025-04-01
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import psutil
from django.conf import settings
from django.db import transaction
from django.utils import timezone
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from news.models import Article

from .models import ProcessingTask

# Setup logger
logger = logging.getLogger(__name__)

# Model names and paths
DEFAULT_BIAS_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"  # Placeholder, would use actual fake news model
MODELS = {
    "distilbert": {"name": "DistilBERT", "path": DEFAULT_BIAS_MODEL, "max_length": 512},
    "tinybert": {
        "name": "TinyBERT",
        "path": "huawei-noah/TinyBERT_General_4L_312D",
        "max_length": 512,
    },
    "mobilebert": {
        "name": "MobileBERT",
        "path": "google/mobilebert-uncased",
        "max_length": 512,
    },
    "albert": {"name": "ALBERT", "path": "albert-base-v2", "max_length": 512},
}

# Cached models
_model_cache = {}


def queue_processing_for_articles(
        task_type: str, articles: Optional[List[int]] = None
) -> int:
    """
    Queue processing tasks for articles.
    """
    if task_type not in [ProcessingTask.SUMMARIZATION, ProcessingTask.BIAS_DETECTION]:
        raise ValueError(f"Invalid task type: {task_type}")

    # Get articles to process
    if articles:
        query = Article.objects.filter(id__in=articles)
    else:
        # Find articles without the specified processing
        if task_type == ProcessingTask.SUMMARIZATION:
            query = Article.objects.filter(summary__isnull=True)
        else:  # BIAS_DETECTION
            query = Article.objects.filter(bias_score__isnull=True)

    # Exclude articles with pending or processing tasks
    query = query.exclude(
        processing_tasks__task_type=task_type,
        processing_tasks__status__in=[
            ProcessingTask.PENDING,
            ProcessingTask.PROCESSING,
        ],
    )

    # Create tasks
    count = 0
    with transaction.atomic():
        for article in query:
            task = ProcessingTask.objects.create(
                article=article, task_type=task_type, status=ProcessingTask.PENDING
            )

            process_article_by_task(task.id)
            count += 1

    return count


def process_article_by_task(task_id: int) -> bool:
    """
    Process an article based on a task record.

    Args:
        task_id: ID of the ProcessingTask to execute

    Returns:
        bool: Success status
    """
    try:
        # Get the task
        task = ProcessingTask.objects.select_related("article").get(id=task_id)

        # Update status to processing
        task.status = ProcessingTask.PROCESSING
        task.save(update_fields=["status", "updated_at"])

        # Process based on task type
        if task.task_type == ProcessingTask.SUMMARIZATION:
            success = summarize_article(task.article)
        else:  # BIAS_DETECTION
            success = detect_bias_in_article(task.article)

        # Update task status
        if success:
            task.status = ProcessingTask.COMPLETED
        else:
            task.status = ProcessingTask.FAILED
            task.error_message = "Processing failed without specific error"

        task.save(update_fields=["status", "error_message", "updated_at"])
        return success

    except ProcessingTask.DoesNotExist:
        logger.error(f"Task {task_id} does not exist")
        return False
    except Exception as e:
        logger.exception(f"Error processing task {task_id}: {str(e)}")

        # Update task as failed if we can
        try:
            task = ProcessingTask.objects.get(id=task_id)
            task.status = ProcessingTask.FAILED
            task.error_message = str(e)
            task.save(update_fields=["status", "error_message", "updated_at"])
        except Exception:
            pass  # If we can't update the task, just continue

        return False


def summarize_article(article: Article) -> bool:
    """
    Generate a summary for an article.

    Args:
        article: Article object to summarize

    Returns:
        bool: Success status
    """
    if not article.content:
        logger.warning(f"Cannot summarize article {article.id}: No content")
        return False

    try:
        # This is a placeholder. In a real implementation, you would:
        # 1. Load a summarization model
        # 2. Generate a summary
        # 3. Save it to the article

        # Simulated summary generation
        content = article.content
        if len(content) <= 100:
            summary = content
        else:
            # Just a simple extractive approach for demonstration
            sentences = content.split(". ")
            summary = ". ".join(sentences[:3]) + "."

        # Update the article
        article.summary = summary
        article.save(update_fields=["summary", "updated_at"])

        logger.info(f"Generated summary for article {article.id}")
        return True

    except Exception as e:
        logger.exception(f"Error summarizing article {article.id}: {str(e)}")
        return False


def detect_bias_in_article(article: Article, model_key: str = "distilbert") -> bool:
    """
    Detect political bias in an article.

    Args:
        article: Article object to analyze
        model_key: Key of model to use (defaults to distilbert)

    Returns:
        bool: Success status
    """
    if not article.content:
        logger.warning(f"Cannot detect bias in article {article.id}: No content")
        return False

    try:
        # Track performance metrics
        start_time = time.time()
        initial_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB

        # Get detection results
        results = detect_fake_news(article.content, model_key)

        # Calculate resource usage
        processing_time = time.time() - start_time
        current_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
        memory_used = current_memory - initial_memory

        # Save results to article
        article.bias_score = results["bias_score"]
        article.save(update_fields=["bias_score", "updated_at"])

        # Create or update detection result
        from news.models import FakeNewsDetectionResult

        FakeNewsDetectionResult.objects.update_or_create(
            article=article,
            defaults={
                "credibility_score": results["credibility_score"],
                "credibility_category": results["category"],
                "confidence": results["confidence"],
                "model_name": results["model_name"],
                "processing_time": processing_time,
                "explanation": results.get("explanation", ""),
            },
        )

        # Update model metrics
        update_model_metrics(
            model_key, results["confidence"], processing_time, memory_used
        )

        logger.info(f"Detected bias for article {article.id} with model {model_key}")
        return True

    except Exception as e:
        logger.exception(f"Error detecting bias in article {article.id}: {str(e)}")
        return False


def get_model_for_detection(model_key: str = "distilbert"):
    """
    Get the appropriate model for fake news detection.

    Args:
        model_key: Key of model to load

    Returns:
        tuple: (tokenizer, model) for the requested model
    """
    # Check if model is already loaded
    if model_key in _model_cache:
        return _model_cache[model_key]

    # Get model config
    if model_key not in MODELS:
        raise ValueError(f"Unknown model key: {model_key}")

    model_config = MODELS[model_key]
    model_path = model_config["path"]

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    # Cache for future use
    _model_cache[model_key] = (tokenizer, model)

    return tokenizer, model


def detect_fake_news(text: str, model_key: str = "distilbert") -> Dict[str, Any]:
    """
    Detect fake news in the given text using the specified model.

    Args:
        text: Article text to analyze
        model_key: Key of model to use

    Returns:
        dict: Detection results including score, category, confidence
    """
    # This is a simplified placeholder implementation
    # In a real implementation, you would:
    # 1. Preprocess the text
    # 2. Run it through a fine-tuned fake news detection model
    # 3. Return the results

    # For demonstration, we'll use a sentiment classifier as a stand-in
    # In a real implementation, you'd use a model fine-tuned for fake news
    tokenizer, model = get_model_for_detection(model_key)
    classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    # For long texts, we need to chunk and analyze sections
    max_length = MODELS[model_key]["max_length"]

    if len(text) > max_length * 4:
        # Analyze just the beginning, middle, and end for simplicity
        beginning = text[:max_length]
        middle_start = len(text) // 2 - max_length // 2
        middle = text[middle_start : middle_start + max_length]
        end = text[-max_length:]

        samples = [beginning, middle, end]
        results = []

        for sample in samples:
            result = classifier(sample)[0]
            results.append(result)

        # Average the results
        if sum(1 for r in results if r["label"] == "POSITIVE") >= 2:
            label = "POSITIVE"
        else:
            label = "NEGATIVE"

        score = sum(r["score"] for r in results) / len(results)
    else:
        # For shorter texts, analyze the whole thing
        result = classifier(text[:max_length])[0]
        label = result["label"]
        score = result["score"]

    # Map sentiment results to credibility results
    # In a real implementation, this would be the direct output of a fake news model
    credibility_mapping = {"POSITIVE": "mostly_credible", "NEGATIVE": "mostly_fake"}

    # Convert sentiment score to bias score (-1 to 1 range)
    # This is just a placeholder mapping
    bias_score = (score - 0.5) * 2  # Convert 0-1 to -1 to 1

    category = credibility_mapping[label]
    credibility_score = score if label == "POSITIVE" else 1 - score

    return {
        "credibility_score": credibility_score,
        "category": category,
        "confidence": score,
        "bias_score": bias_score,
        "model_name": MODELS[model_key]["name"],
        "explanation": f"This article was analyzed using {MODELS[model_key]['name']}.",
    }


def update_model_metrics(
    model_key: str, confidence: float, processing_time: float, memory_usage: float
):
    """
    Update metrics for a detection model.

    Args:
        model_key: Key of the model
        confidence: Confidence score from detection
        processing_time: Time taken for processing
        memory_usage: Memory used during processing
    """
    from news.models import DetectionModelMetrics

    # This is a simplified implementation that only tracks average metrics
    # A complete implementation would track more detailed performance metrics
    # Get or create metrics record
    try:
        metrics, created = DetectionModelMetrics.objects.get_or_create(
            model_name=MODELS[model_key]["name"],
            defaults={
                "accuracy": 0.85,  # Placeholder values
                "precision_score": 0.86,
                "recall_score": 0.84,
                "f1_score": 0.85,
                "avg_processing_time": processing_time,
                "avg_memory_usage": memory_usage,
                "parameter_count": (
                    66000000 if model_key == "distilbert" else 14500000
                ),  # Placeholder
                "efficiency_score": 0.8,
            },
        )

        if not created:
            # Update running averages
            metrics.avg_processing_time = (metrics.avg_processing_time * 0.9) + (
                processing_time * 0.1
            )
            metrics.avg_memory_usage = (metrics.avg_memory_usage * 0.9) + (
                memory_usage * 0.1
            )
            metrics.save(
                update_fields=["avg_processing_time", "avg_memory_usage", "updated_at"]
            )

    except Exception as e:
        logger.error(f"Error updating model metrics for {model_key}: {str(e)}")
