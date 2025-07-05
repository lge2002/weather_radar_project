# report/views.py

from django.shortcuts import render
import os
from django.conf import settings
from datetime import datetime, timedelta, date, time
import pytz

# --- Import your actual CloudAnalysis model ---
from weather.models import CloudAnalysis 

# --- Image Processing Imports ---
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from sklearn.cluster import KMeans
from shapely.ops import unary_union
import warnings
import io
import base64

warnings.filterwarnings("ignore")

# --- GLOBAL CONFIGURATION FOR IMAGE GENERATION (MUST MATCH YOUR SETUP) ---
SHAPEFILE_PATH = os.path.join(settings.BASE_DIR, 'weather', 'management', 'commands', 'gadm41_IND_2.json') 

# These are the alignment values for the FULL TN map with overlay.
FINAL_MIN_LON = 74.80
FINAL_MAX_LON = 80.37
FINAL_MIN_LAT = 7.98
FINAL_MAX_LAT = 13.53
# -------------------------------------------------------------------------


def report_view(request):
    selected_date_str = request.GET.get('date')
    selected_district = request.GET.get('district', 'All Districts')
    if selected_district == "" or selected_district is None:
        selected_district = "All Districts"

    # --- UPDATED: Get time filter parameters from new frontend names ---
    start_time_hour_str = request.GET.get('start_time_hour')
    start_time_minute_str = request.GET.get('start_time_minute')
    end_time_hour_str = request.GET.get('end_time_hour')
    end_time_minute_str = request.GET.get('end_time_minute')
    # --- END UPDATED ---

    selected_image_view = request.GET.get('image_view_type', '') 

    filter_date = None 
    if selected_date_str:
        try:
            filter_date = datetime.strptime(selected_date_str, '%Y-%m-%d').date()
        except ValueError:
            print(f"Warning: Invalid date format '{selected_date_str}'. Defaulting to today's date.")
            filter_date = date.today() 
    else:
        filter_date = date.today() 
    
    target_date_display_str = filter_date.strftime('%Y-%m-%d')

    # --- Prepare selected time strings for template context (HH:MM format) ---
    selected_start_time_for_template = ''
    selected_end_time_for_template = ''

    # --- Construct datetime objects for actual filtering ---
    filter_start_datetime = None
    filter_end_datetime = None

    current_timezone = pytz.timezone(settings.TIME_ZONE) if settings.USE_TZ else None

    # Logic to set default times if not provided in GET parameters (e.g., initial page load)
    if not start_time_hour_str or not start_time_minute_str:
        now = datetime.now()
        current_minute = now.minute
        current_hour = now.hour

        # Round up minutes to the nearest 15 for 'From Time'                            
        from_minute = (current_minute // 15) * 15
        if current_minute % 15 != 0:
            from_minute = ((current_minute // 15) + 1) * 15
            if from_minute == 60:
                from_minute = 0
                current_hour = (current_hour + 1) % 24

        from_hour = current_hour
        
        # Set default 'From Time'
        start_time_obj = time(from_hour, from_minute)
        selected_start_time_for_template = start_time_obj.strftime("%H:%M")
        filter_start_datetime = datetime.combine(filter_date, start_time_obj)

        # Set default 'To Time' (15 minutes after 'From Time')
        to_minute = from_minute + 15
        to_hour = from_hour
        if to_minute == 60:
            to_minute = 0
            to_hour = (to_hour + 1) % 24
        
        end_time_obj = time(to_hour, to_minute)
        selected_end_time_for_template = end_time_obj.strftime("%H:%M")
        filter_end_datetime = datetime.combine(filter_date, end_time_obj)

    else:
        # If time parameters are provided, use them
        try:
            start_hour = int(start_time_hour_str)
            start_minute = int(start_time_minute_str)
            end_hour = int(end_time_hour_str)
            end_minute = int(end_time_minute_str)

            start_time_obj = time(start_hour, start_minute)
            end_time_obj = time(end_hour, end_minute)

            selected_start_time_for_template = start_time_obj.strftime("%H:%M")
            selected_end_time_for_template = end_time_obj.strftime("%H:%M")

            filter_start_datetime = datetime.combine(filter_date, start_time_obj)
            filter_end_datetime = datetime.combine(filter_date, end_time_obj)

        except ValueError as e:
            print(f"Error parsing time parameters: {e}. Defaulting to full day.")
            # If parsing fails, fall back to full day (00:00 to 23:59:59)
            filter_start_datetime = datetime.combine(filter_date, time(0, 0, 0))
            filter_end_datetime = datetime.combine(filter_date, time(23, 59, 59, 999999))
            selected_start_time_for_template = "00:00"
            selected_end_time_for_template = "23:59"


    # Make filter datetimes timezone-aware if USE_TZ is true
    if current_timezone:
        if filter_start_datetime:
            filter_start_datetime = current_timezone.localize(filter_start_datetime)
        if filter_end_datetime:
            filter_end_datetime = current_timezone.localize(filter_end_datetime)


    # --- DEBUGGING PRINTS ---
    print(f"\n--- Generating Images for Date: {target_date_display_str}, District: {selected_district}, Image View: {selected_image_view} ---")
    print(f"Time Range Filter (Backend): {filter_start_datetime} - {filter_end_datetime}")
    print(f"SHAPEFILE_PATH configured as: {SHAPEFILE_PATH}")
    # --- END DEBUGGING PRINTS ---

    # --- LOGIC TO FIND ALL RELEVANT TIMESTAMPED FOLDERS FOR THE SELECTED DATE AND TIME RANGE ---
    image_base_media_path = settings.MEDIA_ROOT 
    
    # This list will store dictionaries, each containing image URLs for a specific timestamp
    # e.g., [{'timestamp': datetime_obj, 'cropped_tn': 'b64_str', 'masked_coimbatore': 'b64_str', ...}, ...]
    generated_images_for_display = [] 

    gdf = None
    gdf_tn = None

    # Try loading shapefile once
    try:
        if not os.path.exists(SHAPEFILE_PATH):
            raise FileNotFoundError(f"Shapefile not found at {SHAPEFILE_PATH}.")
        gdf = gpd.read_file(SHAPEFILE_PATH)
        gdf_tn = gdf[
            (gdf['NAME_1'].str.strip().str.lower() == 'tamilnadu') |
            (gdf['NAME_1'].str.strip().str.lower() == 'tamil nadu')
        ].to_crs("EPSG:4326")
        print(f"Shapefile loaded successfully.")
    except FileNotFoundError as fnfe:
        print(f"CRITICAL ERROR: Shapefile (for generation) not found: {fnfe}")
    except Exception as e:
        print(f"ERROR loading shapefile: {e}")


    if os.path.exists(image_base_media_path) and os.path.isdir(image_base_media_path) and gdf_tn is not None:
        available_image_timestamps_and_folders = []
        for d in os.listdir(image_base_media_path):
            full_path_to_folder = os.path.join(image_base_media_path, d)
            if os.path.isdir(full_path_to_folder):
                try:
                    # Assuming folder name format is YYYY-MM-DD_HH-MM-SS or similar
                    # We need the full timestamp for comparison
                    folder_timestamp_str = d # e.g., '2023-10-27_10-30-00'
                    folder_datetime_obj = datetime.strptime(folder_timestamp_str, '%Y-%m-%d_%H-%M-%S')
                    
                    # Make folder_datetime_obj timezone-aware for comparison if USE_TZ is true
                    if current_timezone:
                        folder_datetime_obj = current_timezone.localize(folder_datetime_obj)

                    # Only consider folders on the selected date and within the time filter
                    if folder_datetime_obj.date() == filter_date and \
                       filter_start_datetime <= folder_datetime_obj < filter_end_datetime:
                        available_image_timestamps_and_folders.append((folder_datetime_obj, full_path_to_folder))
                except (ValueError, IndexError):
                    # Ignore folders that don't match the expected timestamp format
                    pass
        
        # Sort by timestamp to ensure chronological order
        available_image_timestamps_and_folders.sort(key=lambda x: x[0])

        print(f"Found {len(available_image_timestamps_and_folders)} image folders within the selected time range for {target_date_display_str}.")

        # Generate images for each found timestamped folder
        for timestamp_dt, folder_path in available_image_timestamps_and_folders:
            base_image_path_for_this_timestamp = os.path.join(folder_path, 'cropped', 'tamil_nadu_cropped.png')
            
            img_pil = None
            img_np = None
            
            try:
                img_pil = Image.open(base_image_path_for_this_timestamp).convert("RGB")
                img_np = np.array(img_pil)
                height, width, _ = img_np.shape
                transform = from_bounds(FINAL_MIN_LON, FINAL_MIN_LAT, FINAL_MAX_LON, FINAL_MAX_LAT, width, height)

                current_image_set = {
                    'timestamp': timestamp_dt,
                    'cropped_tn': None,
                    'masked_district': None, # Renamed from masked_coimbatore for clarity
                    'aligned_overlay_tn': None,
                }

                # 1. Generate Cropped Tamil Nadu (Radar Only)
                buffer = io.BytesIO()
                img_pil.save(buffer, format="PNG")
                current_image_set['cropped_tn'] = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode('utf-8')
                buffer.close()

                # 2. Generate Masked District Image
                # This part now always attempts to generate the masked image for the selected district
                # If 'All Districts' is selected, it will default to the full cropped TN image as before
                if selected_district == 'All Districts' or gdf_tn.empty:
                    current_image_set['masked_district'] = current_image_set['cropped_tn']
                else:
                    district_rows_for_name = gdf_tn[gdf_tn['NAME_2'].str.lower() == selected_district.lower()]
                    if not district_rows_for_name.empty:
                        all_district_geometries = district_rows_for_name.geometry.to_list()
                        district_polygon_for_mask = unary_union(all_district_geometries)
                        
                        mask = rasterize(
                            [district_polygon_for_mask],
                            out_shape=(height, width),
                            transform=transform,
                            fill=0,
                            all_touched=True,
                            dtype=np.uint8
                        )
                        mask_boolean = mask.astype(bool)
                        cropped_district_img_np = np.zeros_like(img_np)
                        cropped_district_img_np[mask_boolean] = img_np[mask_boolean]

                        buffer = io.BytesIO()
                        Image.fromarray(cropped_district_img_np).save(buffer, format="PNG")
                        current_image_set['masked_district'] = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode('utf-8')
                        buffer.close()
                    else:
                        print(f"Warning: District '{selected_district}' not found in shapefile for masked image generation at {timestamp_dt}.")
                        current_image_set['masked_district'] = None # Or provide a placeholder/error image

                # 3. Generate Overall TN Map with District Outlines (and highlighted district)
                fig, ax = plt.subplots(figsize=(10, 10))
                ax.imshow(img_np, extent=[FINAL_MIN_LON, FINAL_MAX_LON, FINAL_MIN_LAT, FINAL_MAX_LAT])
                gdf_tn.boundary.plot(ax=ax, edgecolor='black', linewidth=0.5)

                if selected_district != 'All Districts':
                    district_rows_for_name_for_highlight = gdf_tn[gdf_tn['NAME_2'].str.lower() == selected_district.lower()]
                    if not district_rows_for_name_for_highlight.empty:
                        district_rows_for_name_for_highlight.boundary.plot(ax=ax, edgecolor='cyan', linewidth=2, linestyle='--', label=selected_district)
                        ax.set_title(f"Aligned Screenshot with {selected_district} Highlighted ({timestamp_dt.strftime('%H:%M')})")
                        ax.legend()
                    else:
                        ax.set_title(f"Aligned Screenshot (District '{selected_district}' not found for highlight) ({timestamp_dt.strftime('%H:%M')})")
                else:
                    ax.set_title(f"Aligned Screenshot with All TN District Outlines ({timestamp_dt.strftime('%H:%M')})")

                ax.set_xlabel("Longitude")
                ax.set_ylabel("Latitude")
                ax.set_aspect('equal')
                plt.tight_layout()

                buffer = io.BytesIO()
                plt.savefig(buffer, format="PNG", bbox_inches='tight', pad_inches=0.1)
                current_image_set['aligned_overlay_tn'] = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode('utf-8')
                buffer.close()
                plt.close(fig)
                
                generated_images_for_display.append(current_image_set)

            except FileNotFoundError:
                print(f"Warning: Base map image '{base_image_path_for_this_timestamp}' not found for {timestamp_dt}. Skipping this timestamp for images.")
            except Exception as e:
                print(f"An unexpected error occurred during image generation for {timestamp_dt}: {e}")
            finally:
                if 'fig' in locals() and fig: # Ensure figure is closed even on error
                    plt.close(fig)
                
    else:
        print(f"Image generation skipped: Base media directory or shapefile not loaded successfully.")

    # --- Filtering CloudAnalysis data ---
    cloud_analysis_query = CloudAnalysis.objects.filter(
        timestamp__date=filter_date # Filter by date part of timestamp
    )

    if selected_district != 'All Districts':
        cloud_analysis_query = cloud_analysis_query.filter(city__iexact=selected_district)

    # Apply time filters if valid filter_start_datetime and filter_end_datetime were constructed
    if filter_start_datetime and filter_end_datetime:

        cloud_analysis_query = cloud_analysis_query.filter(
            timestamp__gte=filter_start_datetime,
            timestamp__lt=filter_end_datetime # Changed from __lte to __lt for standard interval
        )
    

    EXCLUDE_NO_PRECIP_MESSAGE = "No significant cloud levels found for precipitation"
    cloud_analysis_query = cloud_analysis_query.exclude(values__iexact=EXCLUDE_NO_PRECIP_MESSAGE)


    filtered_cloud_analysis_data = list(cloud_analysis_query.order_by('city', 'timestamp'))
    
    print(f"Fetched {len(filtered_cloud_analysis_data)} weather data points for {target_date_display_str} and {selected_district} (excluding 'no precipitation' values).")


    full_available_districts = []
    try:
        if not os.path.exists(SHAPEFILE_PATH):
            print(f"ERROR: Shapefile for district list not found at {SHAPEFILE_PATH}. Falling back to default list.")
            full_available_districts = ['Coimbatore', 'Chennai', 'Madurai', 'Trichy', 'Salem', 'Ariyalur']
        else:
            temp_gdf = gpd.read_file(SHAPEFILE_PATH)
            gdf_tn_districts_for_list = temp_gdf[
                (temp_gdf['NAME_1'].str.strip().str.lower() == 'tamilnadu') |
                (temp_gdf['NAME_1'].str.strip().str.lower() == 'tamil nadu')
            ]
            
            if 'NAME_2' in gdf_tn_districts_for_list.columns:
                unique_districts = gdf_tn_districts_for_list['NAME_2'].dropna().unique().tolist()
                full_available_districts = sorted(unique_districts)
                print(f"Dynamically loaded {len(full_available_districts)} districts from shapefile.")
            else:
                print("Warning: 'NAME_2' column not found in shapefile for district extraction. Falling back to default list.")
                full_available_districts = ['Coimbatore', 'Chennai', 'Madurai', 'Trichy', 'Salem', 'Ariyalur']

    except Exception as e:
        print(f"Error loading districts from shapefile for dropdown: {e}. Falling back to default list.")
        full_available_districts = ['Coimbatore', 'Chennai', 'Madurai', 'Trichy', 'Salem', 'Ariyalur']


    context = {
        'generated_images_for_display': generated_images_for_display, # Pass the list of image sets
        'selected_date': filter_date.strftime('%Y-%m-%d'),
        'selected_district': selected_district,
        'available_districts': full_available_districts,
        'cloud_analysis_data': filtered_cloud_analysis_data, 
        'selected_image_view': selected_image_view,
        # --- UPDATED: Pass the HH:MM strings to context for frontend dropdowns ---
        'selected_start_time': selected_start_time_for_template,
        'selected_end_time': selected_end_time_for_template,
        # --- END UPDATED ---
    }
    return render(request, 'report/report.html', context)