"""
@file processing/services.py
@brief Services for article processing tasks.

This module provides functionality for article summarization and
bias detection using NLP models.

@author Ameed Othman
@date 2025-03-05
"""

import logging
import time
from typing import Dict, Tuple, Optional, Any, Union
from django.conf import settings
from django.core.cache import cache
from django.db import transaction
from news.models import Article
from .models import (
    ProcessingTask,
    SummarizationResult,
    BiasDetectionResult
)

# Setup logger
logger = logging.getLogger(__name__)

# Constants
CACHE_TIMEOUT = 60 * 60 * 24  # 24 hours
SUMMARIZATION_MODELS = {
    'default': 'bart-large-cnn',
    'fast': 't5-small',
    'multilingual': 'mbart-large-cc25'
}
BIAS_DETECTION_MODELS = {
    'default': 'bias-bert-base',
    'detailed': 'bias-roberta-large'
}


def get_nlp_processor(task_type: str, model_key: str = 'default') -> Any:
    """
    Get the appropriate NLP processor for the task.

    This is a factory function that returns the appropriate processor
    for summarization or bias detection tasks.

    Args:
        task_type (str): Type of task ('summarization' or 'bias_detection')
        model_key (str): Key for model selection

    Returns:
        object: NLP processor instance

    Raises:
        ValueError: If task_type is not recognized
    """
    if task_type == ProcessingTask.SUMMARIZATION:
        return SummarizationProcessor(model_key)
    elif task_type == ProcessingTask.BIAS_DETECTION:
        return BiasDetectionProcessor(model_key)
    else:
        raise ValueError(f"Unknown task type: {task_type}")


class SummarizationProcessor:
    """
    Handles article summarization using transformer models.
    """

    def __init__(self, model_key: str = 'default'):
        """
        Initialize the summarization processor.

        Args:
            model_key (str): Key for model selection from SUMMARIZATION_MODELS
        """
        self.model_name = SUMMARIZATION_MODELS.get(model_key, SUMMARIZATION_MODELS['default'])
        self.model = None
        self.tokenizer = None

    def _load_model(self):
        """
        Lazy-load the transformer model and tokenizer.
        """
        if self.model is None or self.tokenizer is None:
            try:
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

                logger.info(f"Loading summarization model: {self.model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
                logger.info(f"Successfully loaded model: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to load summarization model: {str(e)}")
                raise

    def summarize(self, text: str, max_length: int = 150, min_length: int = 40) -> Dict[str, Any]:
        """
        Generate a summary for the given text.

        Args:
            text (str): Text to summarize
            max_length (int): Maximum token length for summary
            min_length (int): Minimum token length for summary

        Returns:
            dict: Dictionary containing summary text and metadata
        """
        start_time = time.time()

        try:
            self._load_model()

            # Truncate input if too long (model-specific limit)
            max_input_length = 1024
            if len(text.split()) > max_input_length:
                logger.warning(f"Text too long, truncating to {max_input_length} tokens")
                text = ' '.join(text.split()[:max_input_length])

            # Tokenize input
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_input_length)

            # Generate summary
            summary_ids = self.model.generate(
                inputs["input_ids"],
                num_beams=4,
                min_length=min_length,
                max_length=max_length,
                early_stopping=True
            )

            # Decode summary
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

            # Calculate processing time
            processing_time = time.time() - start_time

            result = {
                'summary': summary,
                'model_name': self.model_name,
                'processing_time': processing_time
            }

            return result

        except Exception as e:
            logger.error(f"Summarization error: {str(e)}")
            raise


class BiasDetectionProcessor:
    """
    Handles political bias detection in news articles.
    """

    def __init__(self, model_key: str = 'default'):
        """
        Initialize the bias detection processor.

        Args:
            model_key (str): Key for model selection from BIAS_DETECTION_MODELS
        """
        self.model_name = BIAS_DETECTION_MODELS.get(model_key, BIAS_DETECTION_MODELS['default'])
        self.model = None
        self.tokenizer = None

    def _load_model(self):
        """
        Lazy-load the bias detection model.

        Note: In a real implementation, this would load a fine-tuned
        model for political bias classification.
        """
        if self.model is None or self.tokenizer is None:
            try:
                # In a real implementation, this would load the actual model
                # For this example, we'll simulate model loading
                logger.info(f"Simulating loading bias detection model: {self.model_name}")
                self.model = "simulated_model"
                self.tokenizer = "simulated_tokenizer"
            except Exception as e:
                logger.error(f"Failed to load bias detection model: {str(e)}")
                raise

    def detect_bias(self, text: str) -> Dict[str, Any]:
        """
        Detect political bias in the given text.

        Args:
            text (str): Text to analyze for bias

        Returns:
            dict: Dictionary containing bias score, category, and confidence
        """
        start_time = time.time()

        try:
            self._load_model()

            # Note: This is a placeholder implementation
            # In a real implementation, this would use the loaded model

            # For demonstration purposes, we'll return a mock result
            # that simulates model output
            import random

            # Simulate bias score (-1.0 to 1.0)
            bias_score = random.uniform(-1.0, 1.0)

            # Determine bias category based on score
            if bias_score < -0.6:
                bias_category = BiasDetectionResult.STRONG_LEFT
            elif bias_score < -0.2:
                bias_category = BiasDetectionResult.MODERATE_LEFT
            elif bias_score < 0.2:
                bias_category = BiasDetectionResult.NEUTRAL
            elif bias_score < 0.6:
                bias_category = BiasDetectionResult.MODERATE_RIGHT
            else:
                bias_category = BiasDetectionResult.STRONG_RIGHT

            # Simulate confidence score (higher near extremes, lower near center)
            confidence = 0.5 + 0.4 * abs(bias_score)

            # Calculate processing time
            processing_time = time.time() - start_time

            # Simple explanation based on category
            explanations = {
                BiasDetectionResult.STRONG_LEFT: "Uses language and framing typically associated with strong left-leaning perspectives.",
                BiasDetectionResult.MODERATE_LEFT: "Shows some tendency toward left-leaning narrative and word choice.",
                BiasDetectionResult.NEUTRAL: "Uses balanced language and presents multiple perspectives.",
                BiasDetectionResult.MODERATE_RIGHT: "Shows some tendency toward right-leaning narrative and word choice.",
                BiasDetectionResult.STRONG_RIGHT: "Uses language and framing typically associated with strong right-leaning perspectives."
            }

            result = {
                'bias_score': bias_score,
                'bias_category': bias_category,
                'confidence': confidence,
                'explanation': explanations[bias_category],
                'model_name': self.model_name,
                'processing_time': processing_time
            }

            return result

        except Exception as e:
            logger.error(f"Bias detection error: {str(e)}")
            raise


@transaction.atomic
def process_article(article_id: int, task_type: str, model_key: str = 'default') -> Dict[str, Any]:
    """
    Process an article for summarization or bias detection.

    This function handles the workflow of processing an article,
    updating the task status, and storing results.

    Args:
        article_id (int): ID of the article to process
        task_type (str): Type of processing ('summarization' or 'bias_detection')
        model_key (str): Key for model selection

    Returns:
        dict: Processing result or error information

    Raises:
        Article.DoesNotExist: If article doesn't exist
        ValueError: If invalid task type
    """
    try:
        # Get the article
        article = Article.objects.get(id=article_id)

        # Create or get processing task
        task, created = ProcessingTask.objects.get_or_create(
            article=article,
            task_type=task_type,
            defaults={'status': ProcessingTask.PENDING}
        )

        # If task already exists and is completed, return cached result
        if not created and task.status == ProcessingTask.COMPLETED:
            if task_type == ProcessingTask.SUMMARIZATION:
                try:
                    result = article.summarization_result
                    return {
                        'summary': result.summary_text,
                        'model_name': result.model_name,
                        'processing_time': result.processing_time,
                        'created_at': result.created_at
                    }
                except SummarizationResult.DoesNotExist:
                    pass
            elif task_type == ProcessingTask.BIAS_DETECTION:
                try:
                    result = article.bias_detection_result
                    return {
                        'bias_score': result.bias_score,
                        'bias_category': result.bias_category,
                        'confidence': result.confidence,
                        'explanation': result.explanation,
                        'model_name': result.model_name,
                        'created_at': result.created_at
                    }
                except BiasDetectionResult.DoesNotExist:
                    pass

        # Mark task as processing
        task.mark_as_processing()

        # Check if article has content
        if not article.content:
            error_msg = "Article has no content to process"
            task.mark_as_failed(error_msg)
            return {'error': error_msg}

        # Get appropriate processor
        processor = get_nlp_processor(task_type, model_key)

        # Process based on task type
        if task_type == ProcessingTask.SUMMARIZATION:
            result = processor.summarize(article.content)

            # Store the result
            summary, _ = SummarizationResult.objects.update_or_create(
                article=article,
                defaults={
                    'summary_text': result['summary'],
                    'model_name': result['model_name'],
                    'processing_time': result['processing_time']
                }
            )

            # Update article's summary field
            article.summary = result['summary']
            article.save(update_fields=['summary'])

        elif task_type == ProcessingTask.BIAS_DETECTION:
            result = processor.detect_bias(article.content)

            # Store the result
            bias_result, _ = BiasDetectionResult.objects.update_or_create(
                article=article,
                defaults={
                    'bias_score': result['bias_score'],
                    'bias_category': result['bias_category'],
                    'confidence': result['confidence'],
                    'explanation': result['explanation'],
                    'model_name': result['model_name']
                }
            )

            # Update article's bias_score field
            article.bias_score = result['bias_score']
            article.save(update_fields=['bias_score'])

        # Mark task as completed
        task.mark_as_completed()

        return result

    except Article.DoesNotExist:
        logger.error(f"Article not found: {article_id}")
        raise
    except Exception as e:
        logger.error(f"Error processing article {article_id}: {str(e)}")
        if 'task' in locals():
            task.mark_as_failed(str(e))
        raise


def queue_processing_for_articles(task_type: str, days: int = 1) -> int:
    """
    Queue processing tasks for recent articles.

    Args:
        task_type (str): Type of processing task
        days (int): Process articles from the last N days

    Returns:
        int: Number of articles queued for processing
    """
    from django.utils import timezone
    from datetime import timedelta

    # Get recent articles without the specified processing
    date_threshold = timezone.now() - timedelta(days=days)

    # Query for articles that need processing
    articles_query = Article.objects.filter(
        publication_date__gte=date_threshold
    )

    if task_type == ProcessingTask.SUMMARIZATION:
        # Articles without summary
        articles_query = articles_query.filter(summary__isnull=True)
    elif task_type == ProcessingTask.BIAS_DETECTION:
        # Articles without bias score
        articles_query = articles_query.filter(bias_score__isnull=True)

    # Create tasks for each article
    count = 0
    for article in articles_query:
        _, created = ProcessingTask.objects.get_or_create(
            article=article,
            task_type=task_type,
            defaults={'status': ProcessingTask.PENDING}
        )
        if created:
            count += 1

    logger.info(f"Queued {count} articles for {task_type}")
    return count