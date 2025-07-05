from rest_framework import serializers
from .models import CloudAnalysis

class CloudAnalysisSerializer(serializers.ModelSerializer):
    class Meta:
        model = CloudAnalysis
        fields = '__all__'
