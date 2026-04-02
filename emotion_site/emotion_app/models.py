from django.db import models

class Prediction(models.Model):
    emotion = models.CharField(max_length=50)
    confidence = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.emotion} ({self.confidence}%)"
class Feedback(models.Model):
    prediction = models.ForeignKey(Prediction, on_delete=models.CASCADE)
    rating = models.IntegerField()
    is_accurate = models.BooleanField()
    comment = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Feedback for {self.prediction.emotion}"    