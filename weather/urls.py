from django.urls import path
from .views import CloudAnalysisAPIView

urlpatterns = [
    path('api/cloud/', CloudAnalysisAPIView.as_view(), name='cloud-api'),
]
