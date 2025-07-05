
from django.db import models

class CloudAnalysis(models.Model):
    city = models.CharField(max_length=50)
    values = models.CharField(max_length=255)
    type = models.CharField(max_length=50, default="Weather radar")
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self): 
        return f"{self.city} - {self.values}"
