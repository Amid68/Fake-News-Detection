from django.db import models


class ProcessingTask(models.Model):
    """Model to manage and track article processing tasks."""

    # Task types
    SUMMARIZATION = 'summarization'
    BIAS_DETECTION = 'bias_detection'
    TASK_TYPES = [
        (SUMMARIZATION, 'Summarization'),
        (BIAS_DETECTION, 'Bias Detection'),
    ]

    # Status types
    PENDING = 'pending'
    PROCESSING = 'processing'
    COMPLETED = 'completed'
    FAILED = 'failed'
    STATUS_CHOICES = [
        (PENDING, 'Pending'),
        (PROCESSING, 'Processing'),
        (COMPLETED, 'Completed'),
        (FAILED, 'Failed'),
    ]

    article = models.ForeignKey('news.Article', on_delete=models.CASCADE, related_name='processing_tasks')
    task_type = models.CharField(max_length=20, choices=TASK_TYPES)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default=PENDING)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    error_message = models.TextField(blank=True, null=True)
    priority = models.PositiveSmallIntegerField(default=1, help_text="Priority level (1-5, 1 being highest)")
    processor = models.CharField(max_length=100, blank=True, null=True, help_text="Name of worker processing this task")

    class Meta:
        ordering = ['priority', 'created_at']
        indexes = [
            models.Index(fields=['status']),
            models.Index(fields=['task_type']),
            models.Index(fields=['article']),
        ]

    def __str__(self):
        return f"{self.get_task_type_display()} for Article #{self.article_id} ({self.get_status_display()})"