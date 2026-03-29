from django.contrib import admin
from .models import UserProfile, PredictionHistory

@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ['username', 'email', 'created_at']

@admin.register(PredictionHistory)
class PredictionHistoryAdmin(admin.ModelAdmin):
    list_display = ['username', 'prediction', 'confidence', 'predicted_at']
