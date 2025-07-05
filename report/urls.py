
from django.urls import path
from . import views

app_name = 'report'

urlpatterns = [
    path('', views.report_view, name='report'), 
]