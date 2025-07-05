from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from .models import CloudAnalysis
from .serializers import CloudAnalysisSerializer

class CloudAnalysisAPIView(APIView):
    def get(self, request):
        data = CloudAnalysis.objects.all().order_by('-timestamp')
        serializer = CloudAnalysisSerializer(data, many=True)
        return Response(serializer.data)