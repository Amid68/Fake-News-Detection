from django.core.management.base import BaseCommand
from news.services import fetch_articles_from_api, save_articles_to_db


class Command(BaseCommand):
    help = 'Fetch news articles from configured sources'

    def add_arguments(self, parser):
        parser.add_argument(
            '--source',
            type=str,
            help='Specific source to fetch from',
        )

        parser.add_argument(
            '--category',
            type=str,
            help='Specific category to fetch',
        )

    def handle(self, *args, **options):
        self.stdout.write('Fetching news articles...')

        articles = fetch_articles_from_api(
            source=options.get('source'),
            category=options.get('category')
        )

        if not articles:
            self.stdout.write(self.style.WARNING('No articles fetched!'))
            return

        new_count, updated_count = save_articles_to_db(articles)

        self.stdout.write(
            self.style.SUCCESS(
                f'Successfully fetched {len(articles)} articles. '
                f'Added {new_count} new, updated {updated_count} existing.'
            )
        )