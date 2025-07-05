# weather/management/commands/automate_windy.py

from django.core.management.base import BaseCommand
from django.conf import settings
from weather.models import CloudAnalysis
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
from PIL import Image
from datetime import datetime
import os
import time
import json
from django.template.loader import render_to_string

# CHANGED: Replaced 'weasyprint' import with 'xhtml2pdf'
from xhtml2pdf import pisa # Import pisa for xhtml2pdf

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from shapely.ops import unary_union

import requests
import matplotlib.pyplot as plt # NEW: Added matplotlib import

class Command(BaseCommand):
    help = 'Automates screenshot capture from Windy.com, crops to ALL Tamil Nadu districts, masks with shapefile, and analyzes cloud levels.'

    BLUE_DOT_XPATH = '//*[@id="leaflet-map"]/div[1]/div[4]/div[2]'
    API_ENDPOINT_URL = "http://172.16.7.118:8003/api/tamilnadu/satellite/push.windy_radar_data.php"

    # NEW: Link callback function for xhtml2pdf to handle image paths
    def _link_callback(self, uri, rel):
        """
        Convert HTML URIs to absolute system paths so xhtml2pdf can access them.
        This is crucial for local images specified with 'file:///' protocol.
        """
        # Our images are stored with absolute 'file:///' paths
        if uri.startswith('file:///'):
            # Remove the 'file:///' prefix to get the absolute system path
            return uri[len('file:///'):]
        
        # This part handles Django's static/media URLs, though not strictly needed for your current image paths
        # but good practice for a general link callback.
        sUrl = settings.STATIC_URL        # Typically /static/
        sRoot = settings.STATIC_ROOT      # Absolute path to STATIC_ROOT
        mUrl = settings.MEDIA_URL         # Typically /media/
        mRoot = settings.MEDIA_ROOT       # Absolute path to MEDIA_ROOT

        if uri.startswith(mUrl):
            path = os.path.join(mRoot, uri.replace(mUrl, ""))
        elif uri.startswith(sUrl):
            path = os.path.join(sRoot, uri.replace(sUrl, ""))
        else:
            return uri # Return original URI for other types (e.g., external HTTP links)

        # Ensure the path exists for local files
        if os.path.exists(path):
            return path
        else:
            self.stderr.write(self.style.WARNING(f"Warning: Linked file not found: {path} for URI: {uri}"))
            return uri # Fallback to original URI if path doesn't exist

    # CHANGED: PDF Generation Helper now uses xhtml2pdf
    def _generate_and_save_automation_pdf(self, results_data, current_time, base_folder,
                                           full_screenshot_path_abs, cropped_screenshot_path_abs,
                                           json_output_content):
        """
        Generates a PDF report for a single automation run using xhtml2pdf and saves it to a file.
        """
        pdf_filename = f"automation_report_{current_time.strftime('%Y%m%d_%H%M%S')}.pdf"
        pdf_output_path = os.path.join(base_folder, pdf_filename) # Save in the same timestamped folder

        context = {
            'current_time': current_time,
            'current_run_results': results_data,
            # Ensure these paths are absolute and accessible by the PDF renderer
            'full_screenshot_path_abs': full_screenshot_path_abs,
            'cropped_screenshot_path_abs': cropped_screenshot_path_abs,
            'json_output_content': json_output_content,
        }

        # Render the HTML template
        html_string = render_to_string('weather/automation_report_pdf.html', context)

        # Generate PDF using xhtml2pdf and save to target file
        with open(pdf_output_path, "wb") as pdf_file:
            # CreatePDF returns a status object; check for errors
            pisa_status = pisa.CreatePDF(
                html_string,          # the HTML to convert
                dest=pdf_file,        # file handle to receive result
                link_callback=self._link_callback # Important for local images
            )

        if pisa_status.err:
            raise Exception(f"PDF generation error with xhtml2pdf: {pisa_status.err}")
    # --- END NEW METHOD ---


    def handle(self, **kwargs):
        self.stdout.write(self.style.SUCCESS('Starting Windy.com cloud analysis automation for all Tamil Nadu districts...'))

        shapefile_path = "C:/Users/tamilarasans/Downloads/gadm41_IND_2.json/gadm41_IND_2.json"
        if not os.path.exists(shapefile_path):
            self.stderr.write(self.style.ERROR(f"Critical Error: Shapefile not found at {shapefile_path}. Exiting."))
            return

        try:
            gdf = gpd.read_file(shapefile_path)
            # Ensure 'TamilNadu' is matched robustly, accounting for case/spacing if needed
            # Use .str.strip().str.lower() for robustness
            tamil_nadu_gdf = gdf[
                (gdf['NAME_1'].str.strip().str.lower() == 'tamilnadu') |
                (gdf['NAME_1'].str.strip().str.lower() == 'tamil nadu')
            ]
            
            if tamil_nadu_gdf.empty:
                self.stderr.write(self.style.ERROR("Error: 'TamilNadu' not found in shapefile under 'NAME_1'. Please check the shapefile content."))
                return

            tamil_nadu_gdf = tamil_nadu_gdf.to_crs("EPSG:4326") # Ensure CRS for plotting

            all_tn_districts = tamil_nadu_gdf['NAME_2'].unique().tolist()
            if not all_tn_districts:
                self.stderr.write(self.style.ERROR("Error: No districts found for 'TamilNadu' under 'NAME_2' in shapefile. Exiting."))
                return

            self.stdout.write(f"Found {len(all_tn_districts)} districts in Tamil Nadu: {', '.join(all_tn_districts)}")

        except Exception as e:
            self.stderr.write(self.style.ERROR(f"Error loading or processing shapefile initially: {e}. Exiting."))
            return

        while True:
            self.stdout.write("\n" + "="*50)
            self.stdout.write("STARTING NEW 15-MINUTE CYCLE: Capturing fresh screenshot and performing initial analysis.")
            self.stdout.write("="*50 + "\n")

            current_time = datetime.now()
            timestamp_str = current_time.strftime('%Y-%m-%d_%H-%M-%S')

            base_folder = os.path.join(settings.BASE_DIR, "images", timestamp_str)
            full_image_folder = os.path.join(base_folder, "full")
            cropped_image_folder = os.path.join(base_folder, "cropped")
            # NEW: Folder for the overlayed TN map
            report_image_folder = os.path.join(base_folder, "report_image") 

            os.makedirs(full_image_folder, exist_ok=True)
            os.makedirs(cropped_image_folder, exist_ok=True)
            os.makedirs(report_image_folder, exist_ok=True) # NEW: Create the report_image folder

            full_screenshot_path = os.path.join(full_image_folder, "windy_map_full.png")
            cropped_screenshot_path = os.path.join(cropped_image_folder, "tamil_nadu_cropped.png")
            # NEW: Path for the overlayed TN map
            overlay_tn_map_path = os.path.join(report_image_folder, f"tamil_nadu_overlay_{timestamp_str}.png")

            CROP_BOX = (551, 170, 1065, 687) # Left, Upper, Right, Lower

            driver = None
            current_run_results = []

            try:
                chrome_options = webdriver.ChromeOptions()
                chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
                chrome_options.add_experimental_option('useAutomationExtension', False)
                chrome_options.add_argument("--window-size=1920,1080")
                # chrome_options.add_argument("--headless") # Uncomment for headless execution
                # chrome_options.add_argument("--disable-gpu")
                # chrome_options.add_argument("--no-sandbox")

                driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

                driver.get("https://www.windy.com/-Weather-radar-radar?radar,10.950,77.500,7")
                self.stdout.write(f"Navigated to Windy.com with radar layer active and offset coordinates.")

                wait = WebDriverWait(driver, 20)

                try:
                    self.stdout.write('Attempting to dismiss cookie consent...')
                    cookie_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, 'button.cc-dismiss, a[aria-label="dismiss cookie message"]')))
                    cookie_button.click()
                    self.stdout.write('Cookie consent dismissed.')
                    time.sleep(1)
                except Exception as e:
                    self.stdout.write(f"Could not find or dismiss cookie consent (might not be present): {e}. Continuing...")
                    pass

                self.stdout.write("Waiting for map to fully load (10 seconds)...")
                time.sleep(10)

                self.stdout.write("Attempting to hide the blue dot using JavaScript injection with confirmed XPath...")

                try:
                    dot_element = wait.until(EC.presence_of_element_located((By.XPATH, self.BLUE_DOT_XPATH)))
                    driver.execute_script("arguments[0].style.display = 'none';", dot_element)
                    self.stdout.write("SUCCESS (Attempted): Dot element's display set to 'none' via JavaScript using confirmed XPath.")
                    self.stdout.write("NOTE: This method is often ineffective for elements drawn on a canvas, the dot may still be visible in the screenshot.")
                    time.sleep(1)
                except Exception as e:
                    self.stdout.write(f"FAILED to hide dot via JavaScript at XPath '{self.BLUE_DOT_XPATH}': {e}.")
                    self.stdout.write("The element might not be present by this XPath, or another issue occurred. Trying fallback interactive methods (ESC key only)...")
                    
                    try:
                        self.stdout.write("Fallback: Trying to press ESC key.")
                        ActionChains(driver).send_keys(Keys.ESCAPE).perform()
                        self.stdout.write("Pressed ESC key to dismiss dot.")
                        time.sleep(1)
                    except Exception as esc_e:
                        self.stdout.write(f"Fallback (ESC key) failed: {esc_e}. The dot might still be visible.")
                
                time.sleep(2)

                self.stdout.write(f"Taking full screenshot and saving to: {full_screenshot_path}")
                driver.save_screenshot(full_screenshot_path)
                self.stdout.write("Screenshot saved successfully.")

            except Exception as e:
                self.stderr.write(self.style.ERROR(f"An unexpected error occurred during browser automation: {e}"))
                self.stdout.write("Waiting 15 minutes before retry...\n")
                time.sleep(900)
                continue
            finally:
                if driver:
                    driver.quit()
                    self.stdout.write("Browser closed.")

            # --- Image processing and initial analysis for ALL districts (runs once per 15-min cycle) ---
            try:
                image = Image.open(full_screenshot_path).convert("RGB")
                if not (0 <= CROP_BOX[0] < CROP_BOX[2] <= image.width and
                                 0 <= CROP_BOX[1] < CROP_BOX[3] <= image.height):
                    self.stderr.write(self.style.ERROR("CROP_BOX coordinates are out of bounds. Skipping all district analysis for this run."))
                    time.sleep(900)
                    continue

                cropped_image = image.crop(CROP_BOX)
                cropped_image.save(cropped_screenshot_path)
                self.stdout.write(f"Cropped Tamil Nadu image saved at: {cropped_screenshot_path}")

                img_pil = cropped_image.convert("RGB")
                img_np = np.array(img_pil)
                height, width, _ = img_np.shape

                final_min_lon = 74.80
                final_max_lon = 80.37
                final_min_lat = 7.98
                final_max_lat = 13.53

                transform = from_bounds(final_min_lon, final_min_lat, final_max_lon, final_max_lat, width, height)

                # NEW: Generate and save the Tamil Nadu overlayed map
                self.stdout.write(f"Generating Tamil Nadu map with district outlines and saving to: {overlay_tn_map_path}")
                fig, ax = plt.subplots(figsize=(10, 10))
                ax.imshow(img_np, extent=[final_min_lon, final_max_lon, final_min_lat, final_max_lat])
                
                # Ensure tamil_nadu_gdf is correctly loaded and available for plotting
                if tamil_nadu_gdf is not None and not tamil_nadu_gdf.empty:
                    tamil_nadu_gdf.boundary.plot(ax=ax, edgecolor='black', linewidth=0.5)
                    ax.set_title(f"Tamil Nadu Radar with District Outlines ({current_time.strftime('%Y-%m-%d %H:%M:%S')})")
                else:
                    ax.set_title(f"Tamil Nadu Radar (Outlines Not Available) ({current_time.strftime('%Y-%m-%d %H:%M:%S')})")
                    self.stderr.write(self.style.WARNING("Warning: tamil_nadu_gdf not available for plotting outlines."))


                ax.set_xlabel("Longitude")
                ax.set_ylabel("Latitude")
                ax.set_aspect('equal')
                plt.tight_layout()
                plt.savefig(overlay_tn_map_path, bbox_inches='tight', pad_inches=0.1)
                plt.close(fig) # Close the figure to free up memory
                self.stdout.write(self.style.SUCCESS(f"Tamil Nadu overlayed map saved successfully to: {overlay_tn_map_path}"))
                # END NEW

                windy_legend = {
                    (42, 88, 142): "1.5 mm - Blue", (49, 152, 158): "2 mm - Cyan",
                    (58, 190, 140): "3 mm - Aqua Green", (109, 207, 102): "7 mm - Lime",
                    (192, 222, 72): "10 mm - Yellow Green", (241, 86, 59): "20 mm - Red",
                    (172, 64, 112): "30 mm - Purple"
                }

                def match_color_robust(rgb_pixel, legend, max_tolerance=60):
                    best_match_label = None
                    min_distance = float('inf')
                    for legend_color_rgb, label in legend.items():
                        distance = ((rgb_pixel[0] - legend_color_rgb[0])**2 +
                                    (rgb_pixel[1] - legend_color_rgb[1])**2 +
                                    (rgb_pixel[2] - legend_color_rgb[2])**2)**0.5
                        if distance <= max_tolerance and distance < min_distance:
                            min_distance = distance
                            best_match_label = label
                    return best_match_label

                for district_name in all_tn_districts:
                    self.stdout.write(f"\nProcessing district: {district_name} for initial analysis and DB save...")

                    district_masked_folder = os.path.join(base_folder, "masked_cropped", district_name.replace(" ", "_"))
                    os.makedirs(district_masked_folder, exist_ok=True)
                    masked_cropped_path = os.path.join(district_masked_folder, f"{timestamp_str}_{district_name.lower().replace(' ', '_')}_masked.png")
                    
                    current_district_gdf = tamil_nadu_gdf[tamil_nadu_gdf['NAME_2'] == district_name]

                    if current_district_gdf.empty:
                        self.stderr.write(self.style.WARNING(f"Warning: {district_name} not found in the filtered Tamil Nadu shapefile data. Skipping."))
                        continue

                    district_polygon = unary_union(current_district_gdf.geometry.to_list())

                    mask = rasterize(
                        [district_polygon],
                        out_shape=(height, width),
                        transform=transform,
                        fill=0,
                        all_touched=True,
                        dtype=np.uint8
                    )

                    transparent_image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
                    original_rgba_image = cropped_image.convert("RGBA")
                    original_rgba_np = np.array(original_rgba_image)

                    for y in range(height):
                        for x in range(width):
                            if mask[y, x]:
                                r, g, b, a = original_rgba_np[y, x]
                                if a > 0: 
                                    transparent_image.putpixel((x, y), (r, g, b, 255))

                    transparent_image.save(masked_cropped_path)
                    self.stdout.write(f"Masked image of {district_name} saved at: {masked_cropped_path}")

                    image_district_for_analysis = transparent_image.convert('RGB')
                    pixels_to_analyze = list(image_district_for_analysis.getdata())
                    
                    matched_colors = set()
                    for pixel_color in pixels_to_analyze:
                        if pixel_color != (0, 0, 0): 
                            label = match_color_robust(pixel_color, windy_legend, max_tolerance=60)
                            if label:
                                matched_colors.add(label)

                    timestamp_for_db = current_time 
                    color_text = ", ".join(sorted(matched_colors)) if matched_colors else "No significant cloud levels found for precipitation"
                    self.stdout.write(f"Analysis for {district_name}: {color_text}")

                    try:
                        CloudAnalysis.objects.create(
                            city=district_name,
                            values=color_text,
                            type="Weather radar",
                            timestamp=timestamp_for_db
                        )
                        self.stdout.write(self.style.SUCCESS(f"Cloud analysis for {district_name} saved to database."))
                    except Exception as e:
                        self.stderr.write(self.style.ERROR(f"Error saving {district_name} to Django model: {e}"))

                    district_data_for_post_collection = { 
                        "city": district_name,
                        "values": color_text,
                        "type": "Weather radar",
                        "timestamp": timestamp_for_db.strftime('%Y-%m-%d %H:%M:%S')
                    }
                    current_run_results.append(district_data_for_post_collection)
            
            except Exception as e:
                self.stderr.write(self.style.ERROR(f"Error during initial image processing or shapefile handling for all districts: {e}"))
                self.stdout.write("Waiting 15 minutes before retry...\n")
                time.sleep(900)
                continue


            # --- Save the collected JSON data locally (once per 15-min cycle) ---
            json_filename = f"cloud_analysis_results_{timestamp_str}.json"
            json_output_path = os.path.join(base_folder, json_filename)
            json_output_content = json.dumps(current_run_results, indent=4)
            try:
                with open(json_output_path, "w") as json_file:
                    json_file.write(json_output_content)
                self.stdout.write(self.style.SUCCESS(f"All initial analysis results for this 15-min cycle saved to JSON at: {json_output_path}"))
            except Exception as e:
                self.stderr.write(self.style.ERROR(f"Error saving full cycle JSON file: {e}"))

            # --- Generate and Save PDF Report for this cycle ---
            self.stdout.write("Generating PDF report for this run...")
            try:
                self._generate_and_save_automation_pdf(
                    current_run_results,
                    current_time,
                    base_folder,
                    full_screenshot_path,
                    cropped_screenshot_path,
                    json_output_content
                )
                pdf_output_filename_for_message = f"automation_report_{current_time.strftime('%Y%m%d_%H%M%S')}.pdf"
                pdf_full_path_for_message = os.path.join(base_folder, pdf_output_filename_for_message)
                self.stdout.write(self.style.SUCCESS(f"PDF report generated and saved successfully to: {pdf_full_path_for_message}"))
            except Exception as e:
                self.stderr.write(self.style.ERROR(f"Error generating PDF report for this run: {e}"))

            # --- Remaining Code ---
            num_post_attempts = 3
            post_interval_seconds = 300

            for i in range(num_post_attempts):
                self.stdout.write(f"\n--- URL PUSHING CYCLE {i + 1} of {num_post_attempts} (using data from this 15-min screenshot) ---")
                
                if self.API_ENDPOINT_URL and current_run_results:
                    self.stdout.write(f"Attempting to send ALL analysis data to {self.API_ENDPOINT_URL} via POST (Cycle {i+1})...")
                    
                    headers = {
                        'Content-Type': 'application/json',
                    }

                    try:
                        self.stdout.write(f"Sending JSON payload: {json.dumps(current_run_results, indent=4)}")
                        response = requests.post(self.API_ENDPOINT_URL, json=current_run_results, headers=headers, timeout=30)
                        response.raise_for_status()

                        self.stdout.write(self.style.SUCCESS(f"Data successfully POSTed to {self.API_ENDPOINT_URL} (Cycle {i+1})."))
                        self.stdout.write(f"API Response Status Code: {response.status_code}")
                        try:
                            self.stdout.write(f"API Response JSON: {response.json()}")
                        except json.JSONDecodeError:
                            self.stdout.write(f"API Response Text: {response.text}")
                    except requests.exceptions.HTTPError as http_err:
                        self.stderr.write(self.style.ERROR(f"HTTP error during POST request (Cycle {i+1}): {http_err}"))
                        if http_err.response:
                            self.stderr.write(self.style.ERROR(f"Response from API (Cycle {i+1}): {http_err.response.text}"))
                    except requests.exceptions.ConnectionError as conn_err:
                        self.stderr.write(self.style.ERROR(f"Connection error during POST request (Cycle {i+1}, Is the server at {self.API_ENDPOINT_URL} reachable and port open?): {conn_err}"))
                    except requests.exceptions.Timeout as timeout_err:
                        self.stderr.write(self.style.ERROR(f"Timeout error during POST request (Cycle {i+1}, API took too long to respond): {timeout_err}"))
                    except requests.exceptions.RequestException as req_err:
                        self.stderr.write(self.style.ERROR(f"An unexpected error occurred during POST request (Cycle {i+1}): {req_err}"))
                else:
                    if not self.API_ENDPOINT_URL:
                        self.stdout.write(self.style.WARNING(f"API_ENDPOINT_URL is not set. Skipping POST request (Cycle {i+1})."))
                    if not current_run_results:
                        self.stdout.write(self.style.WARNING(f"No analysis results to send. Skipping POST request (Cycle {i+1})."))
                
                if i < num_post_attempts - 1:
                    self.stdout.write(f"Inner loop (URL Pushing): Waiting {post_interval_seconds // 60} minutes before next URL push (Cycle {i+2})...\n")
                    time.sleep(post_interval_seconds)
            self.stdout.write("\nFinished all URL pushing cycles for this 15-minute data set.")
            self.stdout.write("Waiting 15 minutes before starting a new full run (fresh screenshot and analysis)....\n")
            time.sleep(900)