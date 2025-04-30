# =============================================================================
# News Aggregator Project Makefile
# =============================================================================
#
# A collection of shortcuts for common Docker and Django operations.
# This Makefile provides simple commands for building, running, and
# managing the news aggregator application in Docker containers.
# =============================================================================

.PHONY: build up down restart logs shell migrate static collectstatic fetch-news process-articles test

# Build and start all services
build:
	docker-compose build

# Start all services
up:
	docker-compose up -d

# Stop all services
down:
	docker-compose down

# Restart services
restart:
	docker-compose restart

# View logs
logs:
	docker-compose logs -f

# Shell into web container
shell:
	docker-compose exec web bash

# Run migrations
migrate:
	docker-compose exec web python manage.py migrate

# Collect static files
collectstatic:
	docker-compose exec web python manage.py collectstatic --noinput

# Fetch news articles
fetch-news:
	docker-compose exec web python manage.py fetch_news --limit 50 --process

# Process articles
process-articles:
	docker-compose exec web python manage.py process_articles --type bias_detection

# Run tests
test:
	docker-compose exec web python manage.py test

# Create a superuser
createsuperuser:
	docker-compose exec web python manage.py createsuperuser

# Backup database
backup-db:
	mkdir -p backups
	docker-compose exec db pg_dump -U postgres news_aggregator > backups/backup-$(shell date +%Y%m%d%H%M%S).sql

# Restore database from latest backup
restore-db:
	@LATEST=$$(ls -t backups/*.sql | head -1); \
	if [ -z "$$LATEST" ]; then \
		echo "No backup files found"; \
		exit 1; \
	fi; \
	echo "Restoring from $$LATEST"; \
	docker-compose exec -T db psql -U postgres news_aggregator < "$$LATEST"

# Deploy to production (assumes proper environment setup)
deploy-prod:
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Clean all containers and volumes (DANGEROUS)
clean-all:
	docker-compose down -v
	docker system prune -f