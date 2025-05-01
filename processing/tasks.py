"""
@file processing/tasks.py
@brief Celery tasks for asynchronous processing of news articles.

This module defines Celery tasks for article processing operations.

@author Ameed Othman
@date 2025-04-01
"""

import logging

from celery import shared_task

# Setup logger
logger = logging.getLogger(__name__)


@shared_task
def process_article_task(task_id):
    """
    Celery task to process an article based on a task record.

    Args:
        task_id: ID of the ProcessingTask to execute

    Returns:
        bool: Success status
    """
    logger.info(f"Starting processing task {task_id}")

    # Import here to avoid circular import
    from .services import process_article_by_task

    result = process_article_by_task(task_id)
    logger.info(f"Completed processing task {task_id} with result: {result}")
    return result
