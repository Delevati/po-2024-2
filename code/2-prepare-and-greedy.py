import os
import re
import zipfile
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path
import geopandas as gpd
from shapely.geometry import Polygon, box, mapping
import rasterio
import numpy as np
import json
from rasterio.warp import transform_bounds
from rasterio import features
import xml.etree.ElementTree as ET
import logging
import traceback
from collections import defaultdict

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Use Pathlib for paths
BASE_VOLUME = Path("/Volumes/luryand")
ZIP_SOURCE_DIR = BASE_VOLUME / "" 
OUTPUT_BASE_DIR = BASE_VOLUME / "coverage_otimization"
TEMP_EXTRACT_DIR = BASE_VOLUME / "temp_extract"
AOI_SHAPEFILE = Path("/Users/luryand/Documents/encode-image/coverage_otimization/code/APA-input/MULTI-POLYGON.shp")
METADATA_DIR = OUTPUT_BASE_DIR / "metadata"

# Thresholds and Parameters
VALID_DATA_THRESHOLD = 0.3
CENTRAL_IMAGE_EFFECTIVE_COVERAGE_THRESHOLD = 0.3
COMPLEMENT_IMAGE_MIN_GEOGRAPHIC_COVERAGE_THRESHOLD = 0.07
MOSAIC_TIME_WINDOW_DAYS = 4
MAX_CLOUD_COVERAGE_THRESHOLD = 0.5
OVERLAP_QUALITY_WEIGHT = 0.3

# --- Initialization ---
OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
TEMP_EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)

# --- Helper Functions ---
def safe_extract(zip_ref, patterns, extract_path):
    extracted_files = {pattern: [] for pattern in patterns}
    try:
        for file_info in zip_ref.infolist():
            if file_info.is_dir(): continue
            filename_part = Path(file_info.filename).name
            for pattern in patterns:
                if pattern in filename_part:
                    target_path = extract_path / filename_part
                    try:
                        with zip_ref.open(file_info) as source, open(target_path, "wb") as target:
                            shutil.copyfileobj(source, target)
                        extracted_files[pattern].append(target_path)
                        break
                    except Exception as e_extract_single:
                        logging.error(f"Error extracting single file {file_info.filename} to {target_path}: {e_extract_single}")
                        if target_path.exists(): target_path.unlink()
                        break
    except zipfile.BadZipFile:
        logging.error(f"Bad zip file: {zip_ref.filename}")
        return None
    except Exception as e:
        logging.error(f"General error extracting from {zip_ref.filename}: {e}")
        return None
    return extracted_files

def remove_dir_contents(path: Path):
    if path.exists() and path.is_dir():
        for item in path.iterdir():
            try:
                if item.is_dir(): shutil.rmtree(item)
                else: item.unlink()
            except Exception as e:
                logging.warning(f"Could not remove {item}: {e}")

def get_date_from_xml(xml_path: Path) -> datetime | None:
    date_formats = ['%Y-%m-%dT%H:%M:%S.%fZ', '%Y-%m-%dT%H:%M:%S.%f', '%Y-%m-%dT%H:%M:%SZ']
    date_tags = ['DATATAKE_SENSING_START', 'SENSING_TIME', 'PRODUCT_START_TIME', 'GENERATION_TIME']
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for tag_name in date_tags:
            for element in root.findall(f'.//*{tag_name}'):
                 if element.text:
                    date_str = element.text.strip()
                    for fmt in date_formats:
                        try: return datetime.strptime(date_str, fmt)
                        except ValueError: continue
                    try: return datetime.strptime(date_str.split('T')[0], '%Y-%m-%d')
                    except ValueError: pass
    except ET.ParseError: logging.warning(f"Could not parse XML file: {xml_path}")
    except Exception as e: logging.warning(f"Error reading date from XML {xml_path}: {e}")
    try:
        match = re.search(r'_(\d{8})T(\d{6})_', xml_path.name)
        if match: return datetime.strptime(f"{match.group(1)}T{match.group(2)}", "%Y%m%dT%H%M%S")
        match = re.search(r'_(\d{8})_', xml_path.name)
        if match: return datetime.strptime(match.group(1), "%Y%m%d")
    except Exception as e: logging.warning(f"Could not extract date from filename {xml_path.name}: {e}")
    return None

def extract_orbit_from_filename(filename: str) -> int | None:
    orbit_match = re.search(r'_R(\d{3})_', filename)
    return int(orbit_match.group(1)) if orbit_match else None

def calculate_coverage_metrics(raster_path: Path, aoi_gdf_crs_raster: gpd.GeoDataFrame, aoi_area_wgs84: float) -> dict:
    metrics = {'geographic_coverage': 0.0, 'valid_pixels_percentage': 0.0, 'effective_coverage': 0.0, 'bounds': None, 'crs': None}
    try:
        with rasterio.open(raster_path) as src:
            metrics['bounds'] = src.bounds
            metrics['crs'] = src.crs if src.crs else None
            aoi_geometry_transformed = aoi_gdf_crs_raster.geometry.iloc[0]
            img_poly_raster_crs = box(*src.bounds)
            img_gdf_raster_crs = gpd.GeoDataFrame(geometry=[img_poly_raster_crs], crs=src.crs)
            try:
                img_gdf_wgs84 = img_gdf_raster_crs.to_crs("EPSG:4326")
                img_poly_wgs84 = img_gdf_wgs84.geometry.iloc[0]
                aoi_geometry_wgs84 = aoi_gdf_crs_raster.to_crs("EPSG:4326").geometry.iloc[0]
            except Exception as e_crs_wgs84:
                logging.error(f"Failed to transform image/AOI bounds to WGS84 for geo coverage calculation: {e_crs_wgs84}")
                return metrics
            if not aoi_geometry_wgs84.intersects(img_poly_wgs84): return metrics
            intersection_wgs84 = aoi_geometry_wgs84.intersection(img_poly_wgs84)
            geo_coverage = intersection_wgs84.area / aoi_area_wgs84 if aoi_area_wgs84 > 0 else 0.0
            metrics['geographic_coverage'] = geo_coverage
            try:
                mask = features.geometry_mask([aoi_geometry_transformed], out_shape=src.shape, transform=src.transform, invert=True, all_touched=True)
                raster_data = src.read(1, masked=False)
                masked_data = raster_data[mask]
                if masked_data.size == 0:
                    logging.warning(f"Masking resulted in zero pixels for {raster_path} within AOI. Check CRS and overlap.")
                    valid_pixels_percentage = 0.0
                else:
                    valid_pixel_count = np.sum(masked_data > 0)
                    valid_pixels_percentage = valid_pixel_count / masked_data.size
                metrics['valid_pixels_percentage'] = valid_pixels_percentage
                metrics['effective_coverage'] = geo_coverage * valid_pixels_percentage
            except ValueError as e_mask:
                logging.warning(f"Masking error for {raster_path} with AOI: {e_mask}. Likely AOI outside raster bounds. Valid pixels set to 0.")
                metrics['valid_pixels_percentage'] = 0.0
                metrics['effective_coverage'] = 0.0
            except Exception as e_valid_pix:
                 logging.error(f"Error calculating valid pixels for {raster_path}: {e_valid_pix}\n{traceback.format_exc()}")
                 metrics['valid_pixels_percentage'] = 0.0
                 metrics['effective_coverage'] = 0.0
    except rasterio.RasterioIOError as e_rio: logging.error(f"Rasterio error opening {raster_path}: {e_rio}")
    except Exception as e: logging.error(f"Unexpected error calculating coverage for {raster_path}: {e}\n{traceback.format_exc()}")
    return metrics

def calculate_cloud_coverage(cloud_mask_path: Path, aoi_gdf_crs_mask: gpd.GeoDataFrame) -> float:
    try:
        with rasterio.open(cloud_mask_path) as cloud_src:
            aoi_geometry_transformed = aoi_gdf_crs_mask.geometry.iloc[0]
            try:
                mask = features.geometry_mask([aoi_geometry_transformed], out_shape=cloud_src.shape, transform=cloud_src.transform, invert=True, all_touched=True)
                cloud_data = cloud_src.read(1, masked=False)
                cloud_data_aoi = cloud_data[mask]
                if cloud_data_aoi.size == 0:
                    logging.warning(f"AOI mask resulted in zero pixels for cloud mask {cloud_mask_path}. Assuming 0% cloud in intersection.")
                    return 0.0
                cloudy_pixels = np.sum(cloud_data_aoi > 30)
                return cloudy_pixels / cloud_data_aoi.size
            except ValueError:
                logging.warning(f"AOI geometry likely outside bounds of cloud mask {cloud_mask_path}. Assuming 100% cloud for safety.")
                return 1.0
            except Exception as e_cloud_mask:
                 logging.error(f"Error during cloud mask processing for {cloud_mask_path}: {e_cloud_mask}\n{traceback.format_exc()}")
                 return 1.0
    except rasterio.RasterioIOError as e_rio: logging.error(f"Rasterio error opening cloud mask {cloud_mask_path}: {e_rio}")
    except Exception as e: logging.error(f"Error calculating cloud cover for {cloud_mask_path}: {e}")
    return 1.0

def check_image_suitability(geo_coverage: float, valid_pix_perc: float, eff_coverage: float, cloud_perc: float) -> tuple[bool, str]:
    if valid_pix_perc <= 1e-6: return False, f"IMAGEM SEM PIXELS VÁLIDOS NA AOI ({valid_pix_perc:.2%})"
    if geo_coverage < COMPLEMENT_IMAGE_MIN_GEOGRAPHIC_COVERAGE_THRESHOLD: return False, f"IMAGEM COM COBERTURA GEOGRÁFICA INSUFICIENTE ({geo_coverage:.2%} < {COMPLEMENT_IMAGE_MIN_GEOGRAPHIC_COVERAGE_THRESHOLD:.0%})"
    min_effective_coverage_required = COMPLEMENT_IMAGE_MIN_GEOGRAPHIC_COVERAGE_THRESHOLD * 0.5
    if eff_coverage < min_effective_coverage_required: return False, f"IMAGEM COM COBERTURA EFETIVA INSUFICIENTE ({eff_coverage:.2%} < {min_effective_coverage_required:.0%})"
    if cloud_perc > MAX_CLOUD_COVERAGE_THRESHOLD: return False, f"IMAGEM REJEITADA: Muitas nuvens ({cloud_perc:.1%} > {MAX_CLOUD_COVERAGE_THRESHOLD:.0%})"
    return True, "OK"

def save_classification_metadata(output_dir: Path, classification: str, metrics: dict, date_obj: datetime | None, orbit: int | None, zip_filename: str):
    bounds_data = None
    if metrics.get('bounds'):
        b = metrics['bounds']
        bounds_data = {'left': b.left, 'bottom': b.bottom, 'right': b.right, 'top': b.top}
    metadata = {
        'source_zip': zip_filename,
        'filename': Path(zip_filename).stem,
        'status': metrics.get('status', 'unknown'),
        'reason': metrics.get('reason', ''),
        'class': classification if metrics.get('status', 'error').startswith('accepted') else None,
        'date': date_obj.isoformat() if date_obj else None,
        'orbit': orbit,
        'geographic_coverage': metrics.get('geographic_coverage', 0.0),
        'valid_pixels_percentage': metrics.get('valid_pixels_percentage', 0.0),
        'effective_coverage': metrics.get('effective_coverage', 0.0),
        'cloud_coverage': metrics.get('cloud_coverage', 1.0),
        'bounds': bounds_data,
        'crs': str(metrics.get('crs')) if metrics.get('crs') else None,
        'tci_path': metrics.get('tci_path'),
        'cloud_mask_path': metrics.get('cloud_mask_path'),
        'temp_tci_path': metrics.get('temp_tci_path'),
        'temp_cloud_mask_path': metrics.get('temp_cloud_mask_path')
    }
    try:
        meta_filename = output_dir / f"{Path(zip_filename).stem}_metadata.json"
        with open(meta_filename, 'w') as f: json.dump(metadata, f, indent=2)
    except IOError as e: logging.error(f"  Failed to save metadata for {zip_filename} to {output_dir}: {e}")
    except TypeError as e_serial: logging.error(f"  Serialization error saving metadata for {zip_filename}: {e_serial}")

def classify_image(effective_coverage: float) -> str:
    return "central" if effective_coverage >= CENTRAL_IMAGE_EFFECTIVE_COVERAGE_THRESHOLD else "complement"

# --- Main Processing Logic ---
def process_single_zip_file(zip_path: Path, index: int, total: int, aoi_gdf_wgs84: gpd.GeoDataFrame, aoi_area_wgs84: float) -> dict | None:
    logging.info(f"Processing {index+1}/{total}: {zip_path.name}")
    temp_dir = TEMP_EXTRACT_DIR / zip_path.stem
    temp_dir.mkdir(exist_ok=True)
    output_dir = None
    result_data = {
        'status': 'error', 'reason': 'Unknown processing error', 'filename': zip_path.name, 'date': None, 'orbit': None,
        'geographic_coverage': 0.0, 'valid_pixels_percentage': 0.0, 'effective_coverage': 0.0, 'cloud_coverage': 1.0,
        'bounds': None, 'crs': None, 'path': None, 'tci_path': None, 'temp_tci_path': None,
        'cloud_mask_path': None, 'temp_cloud_mask_path': None
    }
    required_patterns = {"MTD_MSIL2A.xml": None, "MSK_CLDPRB_20m.jp2": None, "TCI_10m.jp2": None}
    xml_path, cloud_mask_path_temp, rgb_image_path_temp = None, None, None
    try:
        remove_dir_contents(temp_dir)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            extracted_files = safe_extract(zip_ref, list(required_patterns.keys()), temp_dir)
        if not extracted_files:
            result_data.update({'status': 'rejected', 'reason': 'Extraction failed or bad zip'})
            result_data['orbit'] = extract_orbit_from_filename(zip_path.name)
            save_classification_metadata(METADATA_DIR, None, result_data, None, result_data['orbit'], zip_path.name)
            return result_data

        xml_path = extracted_files.get("MTD_MSIL2A.xml", [None])[0]
        cloud_mask_path_temp = extracted_files.get("MSK_CLDPRB_20m.jp2", [None])[0]
        rgb_image_path_temp = extracted_files.get("TCI_10m.jp2", [None])[0]

        if rgb_image_path_temp: result_data['temp_tci_path'] = str(rgb_image_path_temp)
        if cloud_mask_path_temp: result_data['temp_cloud_mask_path'] = str(cloud_mask_path_temp)

        missing_files = [p for p, f in zip(required_patterns.keys(), [xml_path, cloud_mask_path_temp, rgb_image_path_temp]) if not f]
        if missing_files:
            reason = f'Missing required files: {", ".join(missing_files)}'
            logging.warning(f"  {reason} in {zip_path.name}")
            result_data.update({'status': 'rejected', 'reason': reason})
            if xml_path: result_data['date'] = get_date_from_xml(xml_path)
            result_data['orbit'] = extract_orbit_from_filename(zip_path.name)
            save_classification_metadata(METADATA_DIR, None, result_data, result_data['date'], result_data['orbit'], zip_path.name)
            return result_data

        date_obj = get_date_from_xml(xml_path)
        orbit = extract_orbit_from_filename(zip_path.name)
        result_data['date'] = date_obj
        result_data['orbit'] = orbit

        if not date_obj:
            reason = 'Date extraction failed from XML and filename'
            logging.warning(f"  Could not extract date for {zip_path.name}. Skipping.")
            result_data.update({'status': 'rejected', 'reason': reason})
            save_classification_metadata(METADATA_DIR, None, result_data, date_obj, orbit, zip_path.name)
            return result_data

        coverage_metrics = {}
        cloud_percentage = 1.0

        try:
            with rasterio.open(rgb_image_path_temp) as src_rgb:
                target_crs_rgb = src_rgb.crs
            if not target_crs_rgb:
                raise ValueError(f"Could not determine CRS for TCI image: {rgb_image_path_temp}")

            try:
                aoi_gdf_rgb_crs = aoi_gdf_wgs84.to_crs(target_crs_rgb)
            except Exception as e_crs_tci:
                 logging.error(f"Could not transform AOI to TCI CRS for {rgb_image_path_temp}: {e_crs_tci}")
                 result_data.update({'status': 'error', 'reason': f'CRS transform error: {e_crs_tci}'})
                 save_classification_metadata(METADATA_DIR, None, result_data, date_obj, orbit, zip_path.name)
                 return result_data

            coverage_metrics = calculate_coverage_metrics(rgb_image_path_temp, aoi_gdf_rgb_crs, aoi_area_wgs84)

            result_data.update({
                'bounds': coverage_metrics.get('bounds'),
                'crs': str(coverage_metrics.get('crs')) if coverage_metrics.get('crs') else None,
                'geographic_coverage': coverage_metrics.get('geographic_coverage', 0.0),
                'valid_pixels_percentage': coverage_metrics.get('valid_pixels_percentage', 0.0),
                'effective_coverage': coverage_metrics.get('effective_coverage', 0.0)
            })

            with rasterio.open(cloud_mask_path_temp) as src_cld:
                target_crs_cld = src_cld.crs
            if not target_crs_cld:
                raise ValueError(f"Could not determine CRS for Cloud Mask: {cloud_mask_path_temp}")

            try:
                aoi_gdf_cloud_crs = aoi_gdf_rgb_crs if target_crs_cld == target_crs_rgb else aoi_gdf_wgs84.to_crs(target_crs_cld)
            except Exception as e_crs_cld:
                 logging.error(f"Could not transform AOI to Cloud Mask CRS for {cloud_mask_path_temp}: {e_crs_cld}")
                 result_data.update({'status': 'error', 'reason': f'Cloud Mask CRS transform error: {e_crs_cld}'})
                 save_classification_metadata(METADATA_DIR, None, result_data, date_obj, orbit, zip_path.name)
                 return result_data

            cloud_percentage = calculate_cloud_coverage(cloud_mask_path_temp, aoi_gdf_cloud_crs)

        except ValueError as ve:
             reason = f'Coverage/Cloud calculation pre-check error: {ve}'
             logging.error(f"  {reason} for {zip_path.name}. Skipping.")
             result_data.update({'status': 'error', 'reason': reason, 'cloud_coverage': cloud_percentage})
             save_classification_metadata(METADATA_DIR, None, result_data, date_obj, orbit, zip_path.name)
             return result_data
        except Exception as e_cov:
            reason = f'Coverage/Cloud calculation error: {e_cov}'
            logging.error(f"  {reason} for {zip_path.name}. Skipping.\n{traceback.format_exc()}")
            result_data.update({'status': 'error', 'reason': reason, 'cloud_coverage': cloud_percentage})
            save_classification_metadata(METADATA_DIR, None, result_data, date_obj, orbit, zip_path.name)
            return result_data

        result_data['cloud_coverage'] = cloud_percentage

        logging.info(f"  Geo Cov: {result_data['geographic_coverage']:.2%}, Valid Pix: {result_data['valid_pixels_percentage']:.2%}, Eff Cov: {result_data['effective_coverage']:.2%}, Cloud Cov: {result_data['cloud_coverage']:.2%}")

        is_suitable, reason = check_image_suitability(
            result_data['geographic_coverage'], result_data['valid_pixels_percentage'],
            result_data['effective_coverage'], result_data['cloud_coverage']
        )

        if not is_suitable:
            logging.info(f"  Rejected: {reason}")
            result_data.update({'status': 'rejected', 'reason': reason})
            save_classification_metadata(METADATA_DIR, None, result_data, date_obj, orbit, zip_path.name)
            return result_data

        image_class = classify_image(result_data['effective_coverage'])
        logging.info(f"  CLASSIFIED AS: {image_class.upper()}")
        year_month_dir = date_obj.strftime("%y-%b")
        output_dir = OUTPUT_BASE_DIR / "processed_images" / year_month_dir / zip_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        result_data['path'] = str(output_dir)
        final_tci_path = output_dir / f"{zip_path.stem}_{rgb_image_path_temp.name}"
        final_cloud_mask_path = output_dir / f"{zip_path.stem}_{cloud_mask_path_temp.name}"
        final_xml_path = output_dir / f"{zip_path.stem}_{xml_path.name}"
        result_data['tci_path'] = str(final_tci_path)
        result_data['cloud_mask_path'] = str(final_cloud_mask_path)
        result_data['status'] = 'accepted'
        result_data['class'] = image_class
        
        save_classification_metadata(METADATA_DIR, image_class, result_data, date_obj, orbit, zip_path.name)

        copy_status = 'accepted'
        try:
            shutil.copy2(xml_path, final_xml_path)
            shutil.copy2(cloud_mask_path_temp, final_cloud_mask_path)
            shutil.copy2(rgb_image_path_temp, final_tci_path)
            logging.info(f"  Files copied to {output_dir}")
        except Exception as e_copy:
            logging.error(f"  Error copying files for {zip_path.name} to {output_dir}: {e_copy}")
            copy_status = 'accepted_copy_error'
            result_data['reason'] = 'Accepted but failed to copy files'
            result_data['status'] = copy_status
            save_classification_metadata(METADATA_DIR, image_class, result_data, date_obj, orbit, zip_path.name)

        result_data['status'] = copy_status
        return result_data

    except Exception as e:
        logging.error(f"  Critical error processing {zip_path.name}: {e}\n{traceback.format_exc()}")
        result_data['reason'] = f'Critical error: {e}'
        result_data['status'] = 'error'
        if output_dir and output_dir.exists(): logging.warning(f"  Error occurred, leaving potentially incomplete output directory: {output_dir}")
        return result_data
    finally:
        if temp_dir.exists():
            try: shutil.rmtree(temp_dir)
            except Exception as e_clean: logging.error(f"Failed to remove temp directory tree {temp_dir}: {e_clean}")

# --- Mosaic Finding Functions ---
def find_mosaic_combinations(image_metadata: dict, max_days_diff: int) -> list:
    logging.info("\nAnalyzing potential mosaic combinations...")
    potential_mosaics = []
    centrals = image_metadata.get('central', [])
    complements = image_metadata.get('complement', [])
    all_accepted = centrals + complements
    
    if not all_accepted:
        logging.warning("No accepted central or complement images found to create mosaics.")
        return []

    def calculate_compatibility(base_img, other_img, max_days):
        base_date, other_date = base_img.get('date'), other_img.get('date')
        if not isinstance(base_date, datetime) or not isinstance(other_date, datetime): return None
        days_diff = abs((base_date - other_date).days)
        if days_diff > max_days: return None
        base_orbit, other_orbit = base_img.get('orbit'), other_img.get('orbit')
        orbit_match = base_orbit is not None and base_orbit == other_orbit
        orbit_bonus = 0.05 if orbit_match else 0
        uncovered_area_before = max(0, 1.0 - base_img.get('geographic_coverage', 0))
        overlap_factor = 0.4 if other_img.get('class') == 'central' else 0.2
        contribution_factor = (1.0 - overlap_factor)
        added_geo_coverage = min(uncovered_area_before, other_img.get('geographic_coverage', 0) * contribution_factor)
        estimated_new_geo_coverage = min(1.0, base_img.get('geographic_coverage', 0) + added_geo_coverage)
        quality_factor_other = (1.0 - other_img.get('cloud_coverage', 1.0)) * other_img.get('valid_pixels_percentage', 0.0)
        effectiveness = (added_geo_coverage * quality_factor_other) + orbit_bonus
        return {'image': other_img, 'days_diff': days_diff, 'estimated_coverage_after_add': estimated_new_geo_coverage,
                'quality_factor': quality_factor_other, 'effectiveness_score': effectiveness, 'orbit_match': orbit_match}

    processed_centrals = set()
    for central_img in centrals:
        if central_img['filename'] in processed_centrals: continue

        compatible_images_data = []
        # Check compatibility with complements
        for comp_img in complements:
            comp_data = calculate_compatibility(central_img, comp_img, max_days_diff)
            if comp_data:
                compatible_images_data.append(comp_data)

        # Check compatibility with other centrals
        for other_central in centrals:
            if other_central['filename'] == central_img['filename']: continue
            comp_data = calculate_compatibility(central_img, other_central, max_days_diff)
            if comp_data:
                compatible_images_data.append(comp_data)

        if compatible_images_data:
            compatible_images_data.sort(key=lambda x: x['effectiveness_score'], reverse=True)
            num_additions = min(2, len(compatible_images_data))
            top_additions_data = compatible_images_data[:num_additions]

            mosaic_components = [central_img] + [item['image'] for item in top_additions_data]
            all_filenames = [img['filename'] for img in mosaic_components]
            all_dates = [img['date'] for img in mosaic_components if isinstance(img.get('date'), datetime)]

            if not all_dates:
                continue

            estimated_final_coverage = top_additions_data[0]['estimated_coverage_after_add'] if top_additions_data else central_img['geographic_coverage']
            total_quality = sum([(1.0 - img['cloud_coverage']) * img['valid_pixels_percentage'] for img in mosaic_components])
            avg_quality_factor = total_quality / len(mosaic_components) if mosaic_components else 0

            mosaic = {
                'type': 'mixed_mosaic',
                'base_image': central_img['filename'],
                'component_images': all_filenames,
                'estimated_coverage': estimated_final_coverage,
                'avg_quality_factor': avg_quality_factor,
                'time_window_start': min(all_dates).isoformat(),
                'time_window_end': max(all_dates).isoformat(),
                'component_details': [{'filename': img['filename'], 'class': img['class'], 'date': img['date'].isoformat() if isinstance(img.get('date'), datetime) else None} for img in mosaic_components],
                'contains_sar': True
            }
            potential_mosaics.append(mosaic)

            for img in mosaic_components:
                 if img['class'] == 'central': processed_centrals.add(img['filename'])

    # Logic for complement-only groups (simplified)
    complement_groups = defaultdict(list)
    for comp_img in complements:
        comp_date = comp_img.get('date')
        if not isinstance(comp_date, datetime): continue
        group_key = f"{comp_date.strftime('%Y-%m-%d')}_R{comp_img.get('orbit', 'unknown')}"
        complement_groups[group_key].append(comp_img)

    for group_key, images_in_group in complement_groups.items():
        if len(images_in_group) >= 2:
            images_in_group.sort(key=lambda x: x.get('geographic_coverage', 0.0), reverse=True)
            selected_images = [images_in_group[0]]
            coverage_so_far = images_in_group[0].get('geographic_coverage', 0.0)
            quality_sum = (1.0 - images_in_group[0].get('cloud_coverage', 1.0)) * images_in_group[0].get('valid_pixels_percentage', 0.0)

            for img in images_in_group[1:]:
                contribution_factor = 0.6
                estimated_contribution = img.get('geographic_coverage', 0) * contribution_factor
                potential_new_coverage = min(1.0, coverage_so_far + estimated_contribution)
                if potential_new_coverage > coverage_so_far + 0.05:
                    coverage_so_far = potential_new_coverage
                    selected_images.append(img)
                    quality_sum += (1.0 - img.get('cloud_coverage', 1.0)) * img.get('valid_pixels_percentage', 0.0)
                if coverage_so_far > 0.95: break

            min_combined_coverage_threshold = 0.5
            if coverage_so_far > min_combined_coverage_threshold and len(selected_images) >= 2:
                all_filenames = [img['filename'] for img in selected_images]
                all_dates = [img['date'] for img in selected_images]
                avg_quality_factor = quality_sum / len(selected_images) if selected_images else 0
                mosaic = {
                    'type': 'complement_only',
                    'component_images': all_filenames,
                    'estimated_coverage': coverage_so_far,
                    'avg_quality_factor': avg_quality_factor,
                    'time_window_start': min(all_dates).isoformat(),
                    'time_window_end': max(all_dates).isoformat(),
                    'component_details': [{'filename': img['filename'], 'class': img['class'], 'date': img['date'].isoformat()} for img in selected_images],
                    'contains_sar': True
                }
                potential_mosaics.append(mosaic)

    logging.info(f"Identified {len(potential_mosaics)} potential mosaic combinations.")
    potential_mosaics.sort(key=lambda x: (x.get('estimated_coverage', 0.0), x.get('avg_quality_factor', 0.0)), reverse=True)
    return potential_mosaics

def run_processing_pipeline():
    start_time = time.time()
    logging.info(f"Pipeline started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    stats = defaultdict(int)
    image_metadata = {'central': [], 'complement': []}
    all_processed_metadata = []

    try:
        aoi_gdf_wgs84 = gpd.read_file(AOI_SHAPEFILE)
        if not aoi_gdf_wgs84.crs or aoi_gdf_wgs84.crs.to_epsg() != 4326:
             logging.warning(f"AOI CRS is not EPSG:4326 ({aoi_gdf_wgs84.crs}). Reprojecting...")
             aoi_gdf_wgs84 = aoi_gdf_wgs84.to_crs("EPSG:4326")
        aoi_geometry_union_wgs84 = aoi_gdf_wgs84.union_all()
        aoi_area_wgs84 = aoi_geometry_union_wgs84.area
        aoi_gdf_union_wgs84 = gpd.GeoDataFrame(geometry=[aoi_geometry_union_wgs84], crs="EPSG:4326")
        logging.info(f"AOI loaded from {AOI_SHAPEFILE}. CRS: {aoi_gdf_union_wgs84.crs}. Area (WGS84): {aoi_area_wgs84:.6f} sq. degrees.")
    except Exception as e: logging.error(f"CRITICAL: Failed to load or process AOI file {AOI_SHAPEFILE}: {e}"); return

    zip_files = sorted(list(ZIP_SOURCE_DIR.glob('S2*.zip')))
    total_zip_files = len(zip_files)
    logging.info(f"Found {total_zip_files} ZIP files matching 'S2*.zip' in {ZIP_SOURCE_DIR}")
    if total_zip_files == 0: logging.warning("No ZIP files found to process."); return

    image_process_start = time.time()
    for index, zip_path in enumerate(zip_files):
        stats['processed'] += 1
        result = process_single_zip_file(zip_path, index, total_zip_files, aoi_gdf_union_wgs84, aoi_area_wgs84)
        if result:
            all_processed_metadata.append(result)
            status, reason = result.get('status', 'error'), result.get('reason', 'Unknown status')
            if status.startswith('accepted'):
                stats['accepted'] += 1
                img_class = result.get('class')
                if img_class == 'central': stats['central'] += 1; image_metadata['central'].append(result)
                elif img_class == 'complement': stats['complement'] += 1; image_metadata['complement'].append(result)
                else: logging.warning(f"Accepted image {result.get('filename')} has unknown class: {img_class}")
                if status == 'accepted_copy_error': stats['accepted_copy_error'] += 1
            elif status == 'rejected':
                stats['rejected'] += 1
                if 'Extraction failed' in reason: stats['rejected_extraction'] += 1
                elif 'Missing required files' in reason: stats['rejected_missing_files'] += 1
                elif 'Date extraction failed' in reason: stats['rejected_date'] += 1
                elif 'SEM PIXELS VÁLIDOS' in reason: stats['rejected_no_valid_pixels'] += 1
                elif 'GEOGRÁFICA INSUFICIENTE' in reason: stats['rejected_geo_coverage'] += 1
                elif 'EFETIVA INSUFICIENTE' in reason: stats['rejected_eff_coverage'] += 1
                elif 'Muitas nuvens' in reason: stats['rejected_cloud'] += 1
                else: stats['rejected_unknown'] += 1
            elif status == 'error':
                stats['error'] += 1
                if 'Coverage/Cloud calculation error' in reason: stats['error_coverage_cloud'] += 1
                elif 'CRS' in reason or 'transform' in reason: stats['error_crs'] += 1
                elif 'Critical error' in reason or 'returned None' in reason: stats['error_critical'] += 1
                else: stats['error_unknown'] += 1
            else: stats['error_unknown_status'] += 1
        else:
            logging.error(f"Processing function returned None for {zip_path.name}. Recording as critical error.")
            result = {'status': 'error', 'reason': 'Processing function returned None', 'filename': zip_path.name}
            all_processed_metadata.append(result)
            stats['error'] += 1
            stats['error_critical'] += 1
    image_process_end = time.time()
    image_process_time = image_process_end - image_process_start
    logging.info(f"Image processing completed in {image_process_time:.2f} seconds")

    all_metadata_path = METADATA_DIR / 'all_processed_images_log.json'
    try:
        class DateTimeEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, datetime): return obj.isoformat()
                if isinstance(obj, Path): return str(obj)
                if hasattr(obj, 'left') and hasattr(obj, 'bottom') and hasattr(obj, 'right') and hasattr(obj, 'top'):
                    return {'left': obj.left, 'bottom': obj.bottom, 'right': obj.right, 'top': obj.top}
                return json.JSONEncoder.default(self, obj)
        with open(all_metadata_path, 'w') as f: json.dump(all_processed_metadata, f, indent=2, cls=DateTimeEncoder)
        logging.info(f"Full processing log saved to: {all_metadata_path}")
    except IOError as e: logging.error(f"Failed to save full processing log: {e}")
    except TypeError as e_serial: logging.error(f"Serialization error saving full processing log: {e_serial}")

    greedy_start = time.time()
    logging.info("\n--- Running Greedy Algorithm for Mosaic Combinations ---")
    logging.info(f"Central images available: {len(image_metadata.get('central', []))}")
    logging.info(f"Complement images available: {len(image_metadata.get('complement', []))}")
    good_mosaic_combinations = find_mosaic_combinations(image_metadata, MOSAIC_TIME_WINDOW_DAYS)
    greedy_end = time.time()
    greedy_time = greedy_end - greedy_start
    logging.info(f"Greedy algorithm completed in {greedy_time:.2f} seconds")
    logging.info(f"Found {len(good_mosaic_combinations)} potential mosaic combinations.")

    optimization_params = { 'image_catalog': [], 'mosaic_groups': [] }
    for img_class in ['central', 'complement']:
        for img_meta in image_metadata.get(img_class, []):
             optimization_params['image_catalog'].append({
                 'filename': img_meta['filename'], 'class': img_class, 'date': img_meta['date'].isoformat() if isinstance(img_meta.get('date'), datetime) else None,
                 'orbit': img_meta['orbit'], 'geographic_coverage': img_meta['geographic_coverage'],
                 'effective_coverage': img_meta['effective_coverage'], 'cloud_coverage': img_meta['cloud_coverage'],
                 'quality_factor': (1.0 - img_meta.get('cloud_coverage', 1.0)) * img_meta.get('valid_pixels_percentage', 0.0)
             })
    for idx, mosaic in enumerate(good_mosaic_combinations):
         optimization_params['mosaic_groups'].append({
             'group_id': f'mosaic_{idx+1}', 'type': mosaic['type'], 'images': mosaic['component_images'],
             'estimated_coverage': mosaic['estimated_coverage'], 'quality_factor': mosaic.get('avg_quality_factor', 0.0),
             'time_window_start': mosaic['time_window_start'], 'time_window_end': mosaic['time_window_end'],
             'contains_sar': mosaic.get('contains_sar', False)
         })
    opt_params_path = METADATA_DIR / 'optimization_parameters.json'
    try:
        with open(opt_params_path, 'w') as f: json.dump(optimization_params, f, indent=2)
        logging.info(f"Optimization parameters saved to: {opt_params_path}")
    except IOError as e: logging.error(f"Failed to save optimization parameters: {e}")
    except TypeError as e_serial: logging.error(f"Serialization error saving optimization parameters: {e_serial}")

    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n" + "="*60 + "\nFINAL PROCESSING STATISTICS:\n" + "="*60)
    total_processed, total_accepted, total_rejected, total_errors = stats['processed'], stats['accepted'], stats['rejected'], stats['error']
    print(f"Total ZIP files found:          {total_zip_files}")
    print(f"Total ZIP files processed:      {total_processed}")
    print("-" * 30)
    print(f"Total images accepted:          {total_accepted} ({total_accepted/total_processed*100 if total_processed else 0:.1f}%)")
    print(f"  - Accepted (files copied OK): {total_accepted - stats['accepted_copy_error']}")
    print(f"  - Accepted (file copy error): {stats['accepted_copy_error']}")
    print(f"    - Central images:           {stats['central']}")
    print(f"    - Complement images:        {stats['complement']}")
    print("-" * 30)
    print(f"Total images rejected:          {total_rejected} ({total_rejected/total_processed*100 if total_processed else 0:.1f}%)")
    print("-" * 30)
    print(f"Total processing errors:        {total_errors} ({total_errors/total_processed*100 if total_processed else 0:.1f}%)")
    print("="*60)
    print(f"Potential mosaic combinations identified: {len(good_mosaic_combinations)}")
    print(f"Metadata directory:             {METADATA_DIR}")
    print(f"Optimization parameters file:   {opt_params_path}")
    print(f"This file can now be used with CPLEX for the final optimization step.")
    print("="*60)
    print(f"EXECUTION TIMES:")
    print(f"Image processing time:          {image_process_time:.2f} seconds")
    print(f"Greedy algorithm time:          {greedy_time:.2f} seconds")
    print(f"Total execution time:           {total_time:.2f} seconds")
    print("="*60)
    logging.info(f"Pipeline finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Total execution time: {total_time:.2f} seconds")

if __name__ == "__main__":
    run_processing_pipeline()