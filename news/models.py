from django.db import models

class Source(models.Model):
    name = models.CharField(max_length=255, unique=True)
    base_url = models.TextField()
    api_endpoint = models.TextField(blank=True, null=True)

    def __str__(self):
        return self.name

class Article(models.Model):
    title = models.TextField()
    content = models.TextField(blank=True, null=True)
    source = models.ForeignKey(Source, on_delete=models.CASCADE)
    summary = models.TextField(blank=True, null=True)
    bias_score = models.FloatField(blank=True, null=True)
    publication_date = models.DateTimeField()
    source_article_url = models.TextField(unique=True)

    def __str__(self):
        return self.title