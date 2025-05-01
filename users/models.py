from django.contrib.auth.models import AbstractUser
from django.db import models


class CustomUser(AbstractUser):
    registration_date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.username


class UserPreference(models.Model):
    user = models.ForeignKey(
        CustomUser, on_delete=models.CASCADE, related_name="preferences"
    )
    topic_keyword = models.CharField(max_length=255)

    class Meta:
        unique_together = ("user", "topic_keyword")

    def __str__(self):
        return f"{self.user.username} - {self.topic_keyword}"
