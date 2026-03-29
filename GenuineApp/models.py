from django.db import models

class UserProfile(models.Model):
    username = models.CharField(max_length=100, unique=True)
    password = models.CharField(max_length=200)
    email = models.EmailField()
    contact = models.CharField(max_length=15)
    address = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.username

class PredictionHistory(models.Model):
    username = models.CharField(max_length=100)
    news_text = models.TextField()
    source_url = models.URLField(blank=True, null=True)
    prediction = models.CharField(max_length=50)
    confidence = models.FloatField(default=0.0)
    predicted_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.username} - {self.prediction}"
