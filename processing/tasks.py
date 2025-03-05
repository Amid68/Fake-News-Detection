"""
@file processing/tasks.py
@brief Celery tasks for article processing.

This module defines asynchronous tasks for article summarization and
bias detection using Celery.

@author Ameed Othman
@date 2025-03-05
"""

import logging
from celery import shared_task
from django.db import transaction
from .models import ProcessingTask
from .services import process_article, queue_processing_for_articles

# Setup logger
logger = logging.getLogger(__name__)


@shared_task(bind=True, max_retries=3, default_retry_delay=300)
def process_article_task(self, article_id, task_type, model_key='default'):
    """
    Process an article asynchronously.

    Args:
        article_id (int): ID of the article to process
        task_type (str): Type of processing ('summarization' or 'bias_detection')
        model_key (str): Key for model selection

    Returns:
        dict: Processing result or error information
    """
    try:
        logger.info(f"Processing article {article_id} for {task_type}")

        # Get the task and update task_id
        with transaction.atomic():
            task = ProcessingTask.objects.get(
                article_id=article_id,
                task_type=task_type,
                status__in=[ProcessingTask.PENDING, ProcessingTask.PROCESSING]
            )
            task.mark_as_processing(self.request.id)

        # Process the article
        result = process_article(article_id, task_type, model_key)
        logger.info(f"Successfully processed article {article_id} for {task_type}")

        return result

    except Exception as e:
        logger.error(f"Error in process_article_task for article {article_id}: {str(e)}")

        # Try to mark the task as failed
        try:
            with transaction.atomic():
                task = ProcessingTask.objects.get(
                    article_id=article_id,
                    task_type=task_type
                )
                task.mark_as_failed(str(e))
        except Exception as inner_e:
            logger.error(f"Failed to update task status: {str(inner_e)}")

        # Retry the task if retries are available
        try:
            self.retry(exc=e)
        except self.MaxRetriesExceededError:
            logger.error(f"Max retries exceeded for article {article_id}")

        return {'error': str(e)}


@shared_task
def queue_summarization_tasks(days=1):
    """
    Queue summarization tasks for recent articles.

    Args:
        days (int): Process articles from the last N days

    Returns:
        int: Number of articles queued
    """
    count = queue_processing_for_articles(ProcessingTask.SUMMARIZATION, days)

    # Process tasks as they're queued
    process_queued_tasks.delay(ProcessingTask.SUMMARIZATION)

    return count


@shared_task
def queue_bias_detection_tasks(days=1):
    """
    Queue bias detection tasks for recent articles.

    Args:
        days (int): Process articles from the last N days

    Returns:
        int: Number of articles queued
    """
    count = queue_processing_for_articles(ProcessingTask.BIAS_DETECTION, days)

    # Process tasks as they're queued
    process_queued_tasks.delay(ProcessingTask.BIAS_DETECTION)

    return count


@shared_task
def process_queued_tasks(task_type, batch_size=50):
    """
    Process queued tasks of a specific type.

    Args:
        task_type (str): Type of processing task
        batch_size (int): Number of tasks to process in this batch

    Returns:
        int: Number of tasks processed
    """
    # Get pending tasks
    pending_tasks = ProcessingTask.objects.filter(
        task_type=task_type,
        status=ProcessingTask.PENDING
    ).select_related('article')[:batch_size]

    count = 0
    for task in pending_tasks:
        # Launch processing task
        process_article_task.delay(task.article.id, task_type)
        count += 1

    logger.info(f"Launched {count} {task_type} tasks")
    return count


@shared_task
def retry_failed_tasks(task_type=None, days=7):
    """
    Retry failed processing tasks.

    Args:
        task_type (str, optional): Type of processing task, or None for all types
        days (int): Only retry tasks that failed within the last N days

    Returns:
        int: Number of tasks retried
    """
    from django.utils import timezone
    from datetime import timedelta

    # Calculate cutoff date
    cutoff_date = timezone.now() - timedelta(days=days)

    # Get failed tasks
    query = ProcessingTask.objects.filter(
        status=ProcessingTask.FAILED,
        updated_at__gte=cutoff_date
    )

    if task_type:
        query = query.filter(task_type=task_type)

    count = 0
    for task in query:
        # Reset task status
        task.status = ProcessingTask.PENDING
        task.error_message = None
        task.task_id = None
        task.started_at = None
        task.completed_at = None
        task.save()

        # Queue for processing
        process_article_task.delay(task.article.id, task.task_type)
        count += 1

    logger.info(f"Retried {count} failed tasks")
    return count