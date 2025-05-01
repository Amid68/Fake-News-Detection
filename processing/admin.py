from django.contrib import admin
from django.utils.html import format_html

from .models import ProcessingTask


@admin.register(ProcessingTask)
class ProcessingTaskAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "article_title",
        "task_type",
        "status",
        "priority",
        "created_at",
        "updated_at",
        "actions_column",
    )
    list_filter = ("status", "task_type", "priority")
    search_fields = ("article__title", "error_message")
    readonly_fields = ("created_at", "updated_at")
    date_hierarchy = "created_at"
    actions = ["retry_failed_tasks"]

    def article_title(self, obj):
        """Display the article title with a link to the article admin."""
        if obj.article:
            return format_html(
                '<a href="{}">{}</a>',
                f"/admin/news/article/{obj.article.id}/change/",
                obj.article.title[:50],
            )
        return "N/A"

    article_title.short_description = "Article"

    def actions_column(self, obj):
        """Generate action buttons based on task status."""
        buttons = []

        if obj.status == ProcessingTask.FAILED:
            buttons.append(
                format_html(
                    '<a class="button" href="{}">Retry</a>',
                    f"/admin/processing/processingtask/{obj.id}/retry/",
                )
            )

        return format_html("&nbsp;".join(buttons)) if buttons else "-"

    actions_column.short_description = "Actions"

    def retry_failed_tasks(self, request, queryset):
        """Action to retry failed tasks."""
        from .tasks import process_article_task

        count = 0
        for task in queryset.filter(status=ProcessingTask.FAILED):
            task.status = ProcessingTask.PENDING
            task.error_message = None
            task.save(update_fields=["status", "error_message", "updated_at"])

            # Queue the task for processing
            process_article_task.delay(task.id)
            count += 1

        self.message_user(request, f"Queued {count} tasks for retry.")

    retry_failed_tasks.short_description = "Retry selected failed tasks"
