from django.contrib import admin
from .models import Prediction, Feedback


@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = ('emotion', 'confidence', 'created_at')
    readonly_fields = ('created_at',)


@admin.register(Feedback)
class FeedbackAdmin(admin.ModelAdmin):
    list_display = ('prediction', 'rating', 'is_accurate', 'created_at')
    readonly_fields = ('created_at',)