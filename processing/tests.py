"""
@file processing/tests.py
@brief Test cases for the processing app.

This module provides test cases for models, services, and tasks
in the processing app.

@author Ameed Othman
@date 2025-04-02
"""

from django.test import TestCase
from django.utils import timezone
from django.contrib.auth import get_user_model
from unittest.mock import patch, MagicMock

from news.models import Article, Source, FakeNewsDetectionResult
from .models import ProcessingTask
from .services import queue_processing_for_articles, detect_fake_news

User = get_user_model()


class ProcessingModelTests(TestCase):
    """Tests for processing app models."""

    def setUp(self):
        """Set up test data."""
        # Create a test source
        self.source = Source.objects.create(
            name="Test Source",
            base_url="https://test-source.com"
        )

        # Create a test article
        self.article = Article.objects.create(
            title="Test Article",
            content="This is a test article content.",
            source=self.source,
            publication_date=timezone.now(),
            source_article_url="https://test-source.com/test-article"
        )

    def test_processing_task_creation(self):
        """Test processing task model creation."""
        # Create a processing task
        task = ProcessingTask.objects.create(
            article=self.article,
            task_type=ProcessingTask.BIAS_DETECTION,
            status=ProcessingTask.PENDING
        )

        # Verify task
        self.assertEqual(task.article, self.article)
        self.assertEqual(task.task_type, ProcessingTask.BIAS_DETECTION)
        self.assertEqual(task.status, ProcessingTask.PENDING)
        self.assertTrue(task.created_at is not None)
        self.assertTrue(task.updated_at is not None)
        self.assertIsNone(task.error_message)

        # Test string representation
        self.assertIn("Bias Detection", str(task))
        self.assertIn("Pending", str(task))

    def test_processing_task_status_transition(self):
        """Test processing task status transitions."""
        # Create a processing task
        task = ProcessingTask.objects.create(
            article=self.article,
            task_type=ProcessingTask.BIAS_DETECTION,
            status=ProcessingTask.PENDING
        )

        # Update to processing
        task.status = ProcessingTask.PROCESSING
        task.save()
        task.refresh_from_db()
        self.assertEqual(task.status, ProcessingTask.PROCESSING)

        # Update to completed
        task.status = ProcessingTask.COMPLETED
        task.save()
        task.refresh_from_db()
        self.assertEqual(task.status, ProcessingTask.COMPLETED)

        # Update to failed with error message
        task.status = ProcessingTask.FAILED
        task.error_message = "Test error message"
        task.save()
        task.refresh_from_db()
        self.assertEqual(task.status, ProcessingTask.FAILED)
        self.assertEqual(task.error_message, "Test error message")


class ProcessingServiceTests(TestCase):
    """Tests for processing app services."""

    def setUp(self):
        """Set up test data."""
        # Create a test source
        self.source = Source.objects.create(
            name="Test Source",
            base_url="https://test-source.com"
        )

        # Create test articles
        for i in range(5):
            Article.objects.create(
                title=f"Test Article {i}",
                content=f"This is test article {i} content.",
                source=self.source,
                publication_date=timezone.now(),
                source_article_url=f"https://test-source.com/test-article-{i}"
            )

    @patch('processing.services.process_article_task.delay')
    def test_queue_processing_articles(self, mock_delay):
        """Test queueing articles for processing."""
        # Test queueing all articles for bias detection
        count = queue_processing_for_articles(ProcessingTask.BIAS_DETECTION)

        # Should create 5 tasks
        self.assertEqual(count, 5)
        self.assertEqual(ProcessingTask.objects.count(), 5)
        self.assertEqual(mock_delay.call_count, 5)

        # All tasks should be for bias detection
        for task in ProcessingTask.objects.all():
            self.assertEqual(task.task_type, ProcessingTask.BIAS_DETECTION)
            self.assertEqual(task.status, ProcessingTask.PENDING)

    @patch('processing.services.process_article_task.delay')
    def test_queue_specific_articles(self, mock_delay):
        """Test queueing specific articles for processing."""
        # Get 2 articles
        articles = Article.objects.all()[:2]
        article_ids = [a.id for a in articles]

        # Queue only these articles
        count = queue_processing_for_articles(ProcessingTask.SUMMARIZATION, article_ids)

        # Should create 2 tasks
        self.assertEqual(count, 2)
        self.assertEqual(ProcessingTask.objects.count(), 2)
        self.assertEqual(mock_delay.call_count, 2)

        # Tasks should be for the specific articles
        for task in ProcessingTask.objects.all():
            self.assertEqual(task.task_type, ProcessingTask.SUMMARIZATION)
            self.assertTrue(task.article.id in article_ids)

    @patch('processing.services.get_model_for_detection')
    def test_detect_fake_news(self, mock_get_model):
        """Test fake news detection."""
        # Mock the model and tokenizer
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_get_model.return_value = (mock_tokenizer, mock_model)

        # Mock the pipeline result
        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.return_value = [{'label': 'POSITIVE', 'score': 0.8}]

        with patch('processing.services.pipeline', return_value=mock_pipeline_instance):
            # Test detection
            result = detect_fake_news("This is a test article content.", "distilbert")

            # Check result structure
            self.assertIn('credibility_score', result)
            self.assertIn('category', result)
            self.assertIn('confidence', result)
            self.assertIn('bias_score', result)
            self.assertIn('model_name', result)

            # Positive sentiment should map to 'mostly_credible'
            self.assertEqual(result['category'], 'mostly_credible')
            self.assertAlmostEqual(result['confidence'], 0.8)

            # Test with negative sentiment
            mock_pipeline_instance.return_value = [{'label': 'NEGATIVE', 'score': 0.7}]
            result = detect_fake_news("This is a test article content.", "distilbert")

            # Negative sentiment should map to 'mostly_fake'
            self.assertEqual(result['category'], 'mostly_fake')
            self.assertAlmostEqual(result['confidence'], 0.7)


class ProcessingIntegrationTests(TestCase):
    """Integration tests for processing tasks."""

    def setUp(self):
        """Set up test data."""
        # Create a test source
        self.source = Source.objects.create(
            name="Test Source",
            base_url="https://test-source.com"
        )

        # Create a test article
        self.article = Article.objects.create(
            title="Test Article",
            content="This is a test article with enough content to analyze for bias detection.",
            source=self.source,
            publication_date=timezone.now(),
            source_article_url="https://test-source.com/test-article"
        )

        # Create a processing task
        self.task = ProcessingTask.objects.create(
            article=self.article,
            task_type=ProcessingTask.BIAS_DETECTION,
            status=ProcessingTask.PENDING
        )

    @patch('processing.services.detect_bias_in_article')
    def test_process_article_by_task(self, mock_detect_bias):
        """Test processing an article by task."""
        # Mock the detection function to return success
        mock_detect_bias.return_value = True

        from .services import process_article_by_task

        # Process the task
        result = process_article_by_task(self.task.id)

        # Check result and task status
        self.assertTrue(result)
        self.task.refresh_from_db()
        self.assertEqual(self.task.status, ProcessingTask.COMPLETED)

        # Test with error
        mock_detect_bias.side_effect = Exception("Test error")

        # Create a new task
        task2 = ProcessingTask.objects.create(
            article=self.article,
            task_type=ProcessingTask.BIAS_DETECTION,
            status=ProcessingTask.PENDING
        )

        # Process the new task
        result = process_article_by_task(task2.id)

        # Check result and task status
        self.assertFalse(result)
        task2.refresh_from_db()
        self.assertEqual(task2.status, ProcessingTask.FAILED)
        self.assertEqual(task2.error_message, "Test error")