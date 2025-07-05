# live_datas/views.py
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse # Import JsonResponse
from weather.models import CloudAnalysis
from django.db.models import Max # Import Max for aggregation
import pytz 

def data(request):

    is_ajax = request.headers.get('x-requested-with') == 'XMLHttpRequest' or request.GET.get('format') == 'json'
    latest_timestamps = CloudAnalysis.objects.values('city').annotate(
        latest_timestamp=Max('timestamp')
    ).order_by('city') # Order by city for consistent display order

    # 2. Filter the original queryset to get only the entries matching these latest timestamps
    #    This creates a list of actual CloudAnalysis objects
    latest_cloud_data = []
    for entry in latest_timestamps:
        # Get the specific CloudAnalysis object that has this city and its latest timestamp
        # .first() is used in case multiple entries have the exact same timestamp (though unlikely for "latest")
        record = CloudAnalysis.objects.filter(
            city=entry['city'],
            timestamp=entry['latest_timestamp']
        ).first()
        if record:
            latest_cloud_data.append(record)

    # Sort the final list by timestamp (newest first) for display
    # This ensures the table is ordered consistently for the user
    latest_cloud_data.sort(key=lambda x: x.timestamp, reverse=True)


    if is_ajax:
        # If it's an AJAX request, return data as JSON
        data_for_json = []
        for record in latest_cloud_data:
            # Convert datetime objects to string for JSON serialization
            timestamp_str = record.timestamp.strftime('%Y-%m-%d %H:%M:%S')
            data_for_json.append({
                'city': record.city,
                'values': record.values,
                'type': record.type,
                'timestamp': timestamp_str,
            })
        return JsonResponse({'cloud_analysis_data': data_for_json})
    else:
        # If it's a regular browser request, render the HTML page
        context = {
            'cloud_analysis_data': latest_cloud_data # Pass the fetched latest data
        }
        return render(request, 'live_datas/live.html', context)