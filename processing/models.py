"""
@file models.py
@brief Models for the processing app.

This module defines models related to article processing, including 
summarization and bias detection tasks, logs, and results.

@author Ameed Othman
@date 2025-03-05
"""

from django.db import models
from django.utils.translation import gettext_lazy as _
from django.utils import timezone
from news.models import Article


class ProcessingTask(models.Model):
    """
    Represents a processing task for an article.

    Tracks the status and results of article processing operations like
    summarization and bias detection.
    """

    # Task types
    SUMMARIZATION = 'summarization'
    BIAS_DETECTION = 'bias_detection'
    TASK_TYPE_CHOICES = [
        (SUMMARIZATION, _('Summarization')),
        (BIAS_DETECTION, _('Bias Detection')),
    ]

    # Status options
    PENDING = 'pending'
    PROCESSING = 'processing'
    COMPLETED = 'completed'
    FAILED = 'failed'
    STATUS_CHOICES = [
        (PENDING, _('Pending')),
        (PROCESSING, _('Processing')),
        (COMPLETED, _('Completed')),
        (FAILED, _('Failed')),
    ]

    article = models.ForeignKey(
        Article,
        on_delete=models.CASCADE,
        related_name='processing_tasks'
    )
    task_type = models.CharField(
        max_length=50,
        choices=TASK_TYPE_CHOICES,
        db_index=True
    )
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default=PENDING,
        db_index=True
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    error_message = models.TextField(blank=True, null=True)
    task_id = models.CharField(max_length=255, blank=True, null=True)

    class Meta:
        verbose_name = _('Processing Task')
        verbose_name_plural = _('Processing Tasks')
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['article', 'task_type', 'status']),
        ]

    def __str__(self):
        return f"{self.get_task_type_display()} for {self.article.title[:30]}"

    def mark_as_processing(self, task_id=None):
        """
        Mark task as processing and set started time.

        Args:
            task_id (str, optional): Celery task ID
        """
        self.status = self.PROCESSING
        self.started_at = timezone.now()
        if task_id:
            self.task_id = task_id
        self.save(update_fields=['status', 'started_at', 'task_id', 'updated_at'])

    def mark_as_completed(self):
        """Mark task as completed and set completion time."""
        self.status = self.COMPLETED
        self.completed_at = timezone.now()
        self.save(update_fields=['status', 'completed_at', 'updated_at'])

    def mark_as_failed(self, error_message):
        """
        Mark task as failed with error message.

        Args:
            error_message (str): Description of the error
        """
        self.status = self.FAILED
        self.error_message = error_message
        self.completed_at = timezone.now()
        self.save(update_fields=['status', 'error_message', 'completed_at', 'updated_at'])


class SummarizationResult(models.Model):
    """
    Stores the result of article summarization.
    """

    article = models.OneToOneField(
        Article,
        on_delete=models.CASCADE,
        related_name='summarization_result'
    )
    summary_text = models.TextField()
    model_name = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    processing_time = models.FloatField(
        help_text=_('Processing time in seconds'),
        blank=True,
        null=True
    )
    rating = models.FloatField(
        help_text=_('Quality rating from 0 to 1'),
        blank=True,
        null=True
    )

    class Meta:
        verbose_name = _('Summarization Result')
        verbose_name_plural = _('Summarization Results')
        ordering = ['-created_at']

    def __str__(self):
        return f"Summary for {self.article.title[:30]}"


class BiasDetectionResult(models.Model):
    """
    Stores the result of bias detection for an article.
    """

    # Bias categories
    STRONG_LEFT = 'strong_left'
    MODERATE_LEFT = 'moderate_left'
    NEUTRAL = 'neutral'
    MODERATE_RIGHT = 'moderate_right'
    STRONG_RIGHT = 'strong_right'
    BIAS_CATEGORY_CHOICES = [
        (STRONG_LEFT, _('Strong Left')),
        (MODERATE_LEFT, _('Moderate Left')),
        (NEUTRAL, _('Neutral/Center')),
        (MODERATE_RIGHT, _('Moderate Right')),
        (STRONG_RIGHT, _('Strong Right')),
    ]

    article = models.OneToOneField(
        Article,
        on_delete=models.CASCADE,
        related_name='bias_detection_result'
    )
    bias_score = models.FloatField(
        help_text=_('Score from -1.0 (left) to 1.0 (right)')
    )
    bias_category = models.CharField(
        max_length=50,
        choices=BIAS_CATEGORY_CHOICES
    )
    confidence = models.FloatField(
        help_text=_('Confidence score from 0 to 1')
    )
    explanation = models.TextField(
        help_text=_('Explanation of bias assessment'),
        blank=True
    )
    model_name = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = _('Bias Detection Result')
        verbose_name_plural = _('Bias Detection Results')
        ordering = ['-created_at']

    def __str__(self):
        return f"Bias analysis for {self.article.title[:30]}"

    @property
    def bias_label(self):
        """Human-readable bias category label."""
        return self.get_bias_category_display()