"""
@file news/management/commands/fetch_news.py
@brief Management command to fetch news articles from configured sources.

This command retrieves articles from news sources via APIs and saves
them to the database. It supports filtering by source, category,
and other parameters.

@author Ameed Othman
@date 2025-03-05
"""

import logging
import time
from django.core.management.base import BaseCommand, CommandError
from django.core.cache import cache
from django.db import transaction
from news.services import (
    fetch_articles_from_api,
    save_articles_to_db,
    clear_article_cache,
)
from news.models import Source
from processing.services import queue_processing_for_articles
from processing.models import ProcessingTask

# Setup logger
logger = logging.getLogger(__name__)


class Command(BaseCommand):
    """
    Django management command to fetch news articles from configured sources.
    """

    help = "Fetch news articles from configured sources"

    def add_arguments(self, parser):
        """
        Define command-line arguments for the command.

        Args:
            parser: ArgumentParser instance
        """
        parser.add_argument(
            "--source",
            type=str,
            help="Specific source to fetch from (name or ID)",
        )

        parser.add_argument(
            "--category",
            type=str,
            help="Specific category to fetch (e.g., business, technology)",
        )

        parser.add_argument(
            "--country",
            type=str,
            help="Two-letter country code to fetch news from (e.g., us, gb)",
        )

        parser.add_argument(
            "--limit",
            type=int,
            default=100,
            help="Maximum number of articles to fetch",
        )

        parser.add_argument(
            "--no-cache",
            action="store_true",
            help="Bypass cache and fetch fresh data",
        )

        parser.add_argument(
            "--clear-cache",
            action="store_true",
            help="Clear article cache before fetching",
        )

        parser.add_argument(
            "--process",
            action="store_true",
            help="Queue processing tasks for fetched articles",
        )

    def handle(self, *args, **options):
        """
        Execute the command to fetch news articles.

        Args:
            *args: Additional arguments
            **options: Command options/arguments
        """
        self.stdout.write(self.style.NOTICE("Starting news article fetch..."))

        # Setup variables
        start_time = time.time()
        source_name = options.get("source")
        category = options.get("category")
        country = options.get("country")
        limit = options.get("limit")
        use_cache = not options.get("no_cache")
        process_articles = options.get("process")

        # Clear cache if requested
        if options.get("clear_cache"):
            clear_article_cache()
            self.stdout.write(self.style.SUCCESS("Cleared article cache"))

        # Resolve source if provided
        source_id = None
        if source_name:
            try:
                # Try to get source by name
                source = Source.objects.get(name__iexact=source_name)
                source_id = source.id
                self.stdout.write(f"Using source: {source.name}")
            except Source.DoesNotExist:
                try:
                    # Try to get source by ID
                    source_id = int(source_name)
                    source = Source.objects.get(id=source_id)
                    self.stdout.write(f"Using source: {source.name}")
                except (ValueError, Source.DoesNotExist):
                    raise CommandError(f"Source not found: {source_name}")

        # Fetch articles
        try:
            self.stdout.write(f"Fetching news articles...")

            # Determine page size and number of pages
            page_size = min(limit, 100)  # API might have max page size
            pages = (limit + page_size - 1) // page_size  # Ceiling division

            total_articles = 0
            new_count = 0
            updated_count = 0

            # Fetch articles page by page
            for page in range(1, pages + 1):
                self.stdout.write(f"Fetching page {page}/{pages}...")

                articles = fetch_articles_from_api(
                    source=source_id,
                    category=category,
                    country=country,
                    page_size=page_size,
                    page=page,
                    use_cache=use_cache,
                )

                if not articles:
                    self.stdout.write(
                        self.style.WARNING(f"No articles found for page {page}")
                    )
                    break

                total_articles += len(articles)

                # Save articles to database
                with transaction.atomic():
                    page_new, page_updated = save_articles_to_db(articles)
                    new_count += page_new
                    updated_count += page_updated

                # Break if we've reached our limit
                if total_articles >= limit:
                    break

                # Respect API rate limits with a short delay
                if page < pages:
                    time.sleep(0.5)

            # Queue processing tasks if requested
            if process_articles and (new_count > 0 or updated_count > 0):
                # Queue summarization tasks
                self.stdout.write("Queueing summarization tasks...")
                summ_count = queue_processing_for_articles(ProcessingTask.SUMMARIZATION)

                # Queue bias detection tasks
                self.stdout.write("Queueing bias detection tasks...")
                bias_count = queue_processing_for_articles(
                    ProcessingTask.BIAS_DETECTION
                )

                self.stdout.write(
                    self.style.SUCCESS(
                        f"Queued {summ_count} summarization and {bias_count} bias detection tasks"
                    )
                )

            # Calculate and display execution time
            execution_time = time.time() - start_time

            self.stdout.write(
                self.style.SUCCESS(
                    f"Successfully fetched {total_articles} articles "
                    f"({new_count} new, {updated_count} updated) "
                    f"in {execution_time:.2f} seconds."
                )
            )

        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error fetching articles: {str(e)}"))
            logger.exception("Error in fetch_news command")
            raise CommandError(str(e))
