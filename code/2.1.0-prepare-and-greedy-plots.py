import xml.etree.ElementTree as ET
import re
from datetime import datetime, timedelta
from pathlib import Path
from shapely.geometry import Polygon, box, mapping
from rasterio.warp import transform_bounds, transform
from rasterio import features
from rasterio.enums import Resampling
from matplotlib.patches import Patch
from collections import defaultdict
import geopandas as gpd
import rasterio
import numpy as np
import json
import logging
import shutil
import zipfile
import traceback
import pyproj
from functools import partial
from shapely.ops import transform as shapely_transform
import matplotlib.pyplot as plt
import matplotlib

# --- Configuration ---
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'Arial'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_VOLUME = Path("/Volumes/luryand")
ZIP_SOURCE_DIR = BASE_VOLUME / ""
OUTPUT_BASE_DIR = BASE_VOLUME / "coverage_otimization"
TEMP_EXTRACT_DIR = BASE_VOLUME / "temp_extract"
AOI_SHAPEFILE = Path("/Users/luryand/Documents/encode-image/coverage_otimization/code/APA-input/MULTI-POLYGON.shp")
METADATA_DIR = OUTPUT_BASE_DIR / "metadata"
PLOTS_DIR = OUTPUT_BASE_DIR / 'publication_plots'
VALIDATION_DIR = OUTPUT_BASE_DIR / 'validation'
VALIDATION_TCIS_DIR = VALIDATION_DIR / 'rejected_tcis'
TRASH_DIR = OUTPUT_BASE_DIR / 'trash'

VALID_DATA_THRESHOLD = 0.3
CENTRAL_IMAGE_EFFECTIVE_COVERAGE_THRESHOLD = 0.3
COMPLEMENT_IMAGE_MIN_GEOGRAPHIC_COVERAGE_THRESHOLD = 0.07
MOSAIC_TIME_WINDOW_DAYS = 4
MAX_CLOUD_COVERAGE_THRESHOLD = 0.4
OVERLAP_QUALITY_WEIGHT = 0.3

# --- Initialization ---
OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
TEMP_EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
VALIDATION_TCIS_DIR.mkdir(parents=True, exist_ok=True)
TRASH_DIR.mkdir(parents=True, exist_ok=True)

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
            if not metrics['crs']:
                logging.error(f"CRS is missing for raster {raster_path}. Cannot calculate coverage.")
                return metrics

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

            if not aoi_geometry_wgs84.intersects(img_poly_wgs84):
                logging.debug(f"Image {raster_path.name} does not intersect AOI in WGS84.")
                return metrics

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
    logging.debug(f"Finished coverage calculation for {raster_path.name}. Returning metrics: {metrics}")
    return metrics

def calculate_cloud_coverage(cloud_mask_path: Path, aoi_gdf_crs_mask: gpd.GeoDataFrame) -> float:
    try:
        with rasterio.open(cloud_mask_path) as cloud_src:
            if not cloud_src.crs:
                logging.error(f"CRS is missing for cloud mask {cloud_mask_path}. Cannot calculate cloud coverage.")
                return 1.0

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

def get_cloud_cover_in_geom(img_meta: dict, geometry_wgs84: Polygon) -> float:
    cloud_mask_path_str = img_meta.get('cloud_mask_path') or img_meta.get('temp_cloud_mask_path')
    if not cloud_mask_path_str:
        logging.warning(f"No cloud mask path found for {img_meta.get('filename')} in get_cloud_cover_in_geom.")
        return 1.0
    cloud_mask_path = Path(cloud_mask_path_str)
    if not cloud_mask_path.exists():
        logging.warning(f"Cloud mask file not found at {cloud_mask_path} for {img_meta.get('filename')}.")
        return 1.0

    try:
        with rasterio.open(cloud_mask_path) as cloud_src:
            mask_crs = cloud_src.crs
            if not mask_crs:
                logging.warning(f"CRS missing for cloud mask {cloud_mask_path}. Cannot calculate overlap cloud cover.")
                return 1.0

            try:
                # Transform the geometry from WGS84 to the cloud mask's CRS
                proj_wgs84 = pyproj.Proj('epsg:4326')
                proj_mask = pyproj.Proj(mask_crs)
                transformer_to_mask = pyproj.Transformer.from_proj(proj_wgs84, proj_mask, always_xy=True).transform                
                geometry_mask_crs = shapely_transform(transformer_to_mask, geometry_wgs84)
            except Exception as e_transform:
                logging.error(f"Failed to transform overlap geometry to mask CRS {mask_crs} for {img_meta.get('filename')}: {e_transform}")
                return 1.0

            try:
                # Create a mask for the geometry and extract cloud data within it
                mask = features.geometry_mask([geometry_mask_crs], out_shape=cloud_src.shape, 
                                             transform=cloud_src.transform, invert=True, all_touched=True)
                cloud_data = cloud_src.read(1, masked=False)
                cloud_data_geom = cloud_data[mask]
                
                if cloud_data_geom.size == 0:
                    logging.debug(f"Geometry mask resulted in zero pixels for cloud mask {cloud_mask_path} in get_cloud_cover_in_geom.")
                    return 0.0
                    
                # Count pixels with cloud probability > 30%
                cloudy_pixels = np.sum(cloud_data_geom > 30)
                return cloudy_pixels / cloud_data_geom.size
                
            except ValueError:
                logging.debug(f"Geometry likely outside bounds of cloud mask {cloud_mask_path} in get_cloud_cover_in_geom.")
                return 1.0
            except Exception as e_cloud_mask:
                logging.error(f"Error during cloud mask processing within geometry for {cloud_mask_path}: {e_cloud_mask}")
                return 1.0
    except rasterio.RasterioIOError as e_rio: 
        logging.error(f"Rasterio error opening cloud mask {cloud_mask_path} in get_cloud_cover_in_geom: {e_rio}")
    except Exception as e: 
        logging.error(f"Error calculating cloud cover within geometry for {cloud_mask_path}: {e}")
    
    return 1.0

def check_image_suitability(geo_coverage: float, valid_pix_perc: float, eff_coverage: float, cloud_perc: float) -> tuple[bool, str]:
    if valid_pix_perc <= 1e-6: return False, f"IMAGEM SEM PIXELS VÁLIDOS NA AOI ({valid_pix_perc:.2%})"
    if geo_coverage < COMPLEMENT_IMAGE_MIN_GEOGRAPHIC_COVERAGE_THRESHOLD: return False, f"IMAGEM COM COBERTURA GEOGRÁFICA INSUFICIENTE ({geo_coverage:.2%} < {COMPLEMENT_IMAGE_MIN_GEOGRAPHIC_COVERAGE_THRESHOLD:.0%})"
    min_effective_coverage_required = COMPLEMENT_IMAGE_MIN_GEOGRAPHIC_COVERAGE_THRESHOLD * 0.5
    if eff_coverage < min_effective_coverage_required: return False, f"IMAGEM COM COBERTURA EFETIVA INSUFICIENTE ({eff_coverage:.2%} < {min_effective_coverage_required:.0%})"
    if cloud_perc > MAX_CLOUD_COVERAGE_THRESHOLD: return False, f"IMAGEM REJEITADA: Muitas nuvens ({cloud_perc:.1%} > {MAX_CLOUD_COVERAGE_THRESHOLD:.0%})"
    return True, "OK"

def save_classification_metadata(output_dir: Path, classification: str | None, metrics: dict, date_obj: datetime | None, orbit: int | None, zip_filename: str):
    bounds_data = None
    if metrics.get('bounds'):
        b = metrics['bounds']
        if hasattr(b, 'left') and hasattr(b, 'bottom') and hasattr(b, 'right') and hasattr(b, 'top'):
            bounds_data = {'left': b.left, 'bottom': b.bottom, 'right': b.right, 'top': b.top}
        elif isinstance(b, dict) and all(k in b for k in ['left', 'bottom', 'right', 'top']):
            bounds_data = b # Already a dict
        else:
            logging.warning(f"Unexpected bounds format for {zip_filename}: {type(b)}. Bounds not saved.")

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
        meta_filename = METADATA_DIR / f"{Path(zip_filename).stem}_metadata.json"
        with open(meta_filename, 'w') as f: json.dump(metadata, f, indent=2)
    except IOError as e: logging.error(f"  Failed to save metadata for {zip_filename} to {METADATA_DIR}: {e}")
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
        'cloud_mask_path': None, 'temp_cloud_mask_path': None, 'class': None
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
                logging.debug(f"  AOI successfully transformed to TCI CRS: {target_crs_rgb}")
            except Exception as e_crs_tci:
                 raise ValueError(f"Failed to transform AOI to TCI CRS {target_crs_rgb}: {e_crs_tci}")

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
                logging.debug(f"  AOI successfully transformed to Cloud Mask CRS: {target_crs_cld}")
            except Exception as e_crs_cld:
                 raise ValueError(f"Failed to transform AOI to Cloud Mask CRS {target_crs_cld}: {e_crs_cld}")

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

            temp_tci_path_str = result_data.get('temp_tci_path')
            if temp_tci_path_str and Path(temp_tci_path_str).exists():
                try:
                    tci_filename = Path(temp_tci_path_str).name
                    dest_tci_path_val = VALIDATION_TCIS_DIR / f"{zip_path.stem}_{tci_filename}"
                    VALIDATION_TCIS_DIR.mkdir(parents=True, exist_ok=True)
                    if not dest_tci_path_val.exists():
                         shutil.copy2(temp_tci_path_str, dest_tci_path_val)
                         logging.info(f"  Copied rejected TCI to validation: {dest_tci_path_val.name}")
                    result_data['temp_tci_path'] = str(dest_tci_path_val)
                except Exception as e_copy_val:
                    logging.error(f"  Failed to copy rejected TCI {Path(temp_tci_path_str).name} to validation: {e_copy_val}")
            elif temp_tci_path_str:
                logging.warning(f"  Original temp TCI file not found at {temp_tci_path_str} during rejection step.")
            else:
                logging.warning(f"  No temp_tci_path recorded for rejected image {result_data['filename']}, cannot copy for validation.")

            save_classification_metadata(METADATA_DIR, None, result_data, date_obj, orbit, zip_path.name)
            return result_data

        image_class = classify_image(result_data['effective_coverage'])
        logging.info(f"  CLASSIFIED AS: {image_class.upper()}")
        result_data['class'] = image_class
        result_data['status'] = 'accepted'

        year_month_dir = date_obj.strftime("%y-%b")
        output_dir = OUTPUT_BASE_DIR / "processed_images" / year_month_dir / zip_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        result_data['path'] = str(output_dir)

        final_tci_path = output_dir / rgb_image_path_temp.name
        final_cloud_mask_path = output_dir / cloud_mask_path_temp.name
        final_xml_path = output_dir / xml_path.name

        result_data['tci_path'] = str(final_tci_path)
        result_data['cloud_mask_path'] = str(final_cloud_mask_path)

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
        save_classification_metadata(METADATA_DIR, None, result_data, result_data.get('date'), result_data.get('orbit'), zip_path.name)
        if output_dir and output_dir.exists():
             logging.warning(f"  Error occurred, leaving potentially incomplete output directory: {output_dir}")
        return result_data
    finally:
        if temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
            except Exception as e_clean:
                logging.error(f"Failed to remove temp directory tree {temp_dir}: {e_clean}")

# --- Mosaic Finding Functions ---

def calculate_refined_compatibility(base_img: dict, other_img: dict, max_days: int) -> dict | None:
    # Extract and validate dates
    base_date_str, other_date_str = base_img.get('date'), other_img.get('date')
    try:
        base_date = datetime.fromisoformat(base_date_str) if isinstance(base_date_str, str) else base_date_str
        other_date = datetime.fromisoformat(other_date_str) if isinstance(other_date_str, str) else other_date_str
        if not isinstance(base_date, datetime) or not isinstance(other_date, datetime):
            return None
    except (ValueError, TypeError):
        return None

    # Check time difference
    days_diff = abs((base_date - other_date).days)
    if days_diff > max_days:
        return None

    # Check orbit match
    base_orbit, other_orbit = base_img.get('orbit'), other_img.get('orbit')
    orbit_match = base_orbit is not None and base_orbit == other_orbit
    orbit_bonus = 0.05 if orbit_match else 0

    base_bounds_dict = base_img.get('bounds')
    other_bounds_dict = other_img.get('bounds')
    base_crs_str = base_img.get('crs')
    other_crs_str = other_img.get('crs')
    
    # Verifica se bounds são válidos, senão abre a imagem e pega diretamente
    if not isinstance(base_bounds_dict, dict) or not all(k in base_bounds_dict for k in ['left', 'bottom', 'right', 'top']):
        tci_path = base_img.get('tci_path') or base_img.get('temp_tci_path')
        if tci_path and Path(tci_path).exists():
            try:
                with rasterio.open(tci_path) as src:
                    bounds = src.bounds
                    base_bounds_dict = {
                        'left': bounds.left,
                        'bottom': bounds.bottom,
                        'right': bounds.right,
                        'top': bounds.top
                    }
                    base_crs_str = str(src.crs)
            except Exception:
                return None
        else:
            return None
    
    # Mesmo processo para a outra imagem
    if not isinstance(other_bounds_dict, dict) or not all(k in other_bounds_dict for k in ['left', 'bottom', 'right', 'top']):
        tci_path = other_img.get('tci_path') or other_img.get('temp_tci_path')
        if tci_path and Path(tci_path).exists():
            try:
                with rasterio.open(tci_path) as src:
                    bounds = src.bounds
                    other_bounds_dict = {
                        'left': bounds.left,
                        'bottom': bounds.bottom,
                        'right': bounds.right,
                        'top': bounds.top
                    }
                    other_crs_str = str(src.crs)
            except Exception:
                return None
        else:
            return None
            
    # Continua com a validação de CRS e o restante da função
    if not base_crs_str or not other_crs_str:
        return None

    # Calculate overlap geometry
    overlap_geom_wgs84 = None
    overlap_details = {}
    try:
        base_poly_native = box(base_bounds_dict['left'], base_bounds_dict['bottom'], base_bounds_dict['right'], base_bounds_dict['top'])
        other_poly_native = box(other_bounds_dict['left'], other_bounds_dict['bottom'], other_bounds_dict['right'], other_bounds_dict['top'])

        try:
            proj_base = pyproj.Proj(base_crs_str)
            proj_other = pyproj.Proj(other_crs_str)
            proj_wgs84 = pyproj.Proj('epsg:4326')
        except Exception:
            return None

        transformer_base_to_wgs = pyproj.Transformer.from_proj(proj_base, proj_wgs84, always_xy=True).transform
        transformer_other_to_wgs = pyproj.Transformer.from_proj(proj_other, proj_wgs84, always_xy=True).transform
        
        base_poly_wgs = shapely_transform(transformer_base_to_wgs, base_poly_native)
        other_poly_wgs = shapely_transform(transformer_other_to_wgs, other_poly_native)

        if not base_poly_wgs.is_valid: base_poly_wgs = base_poly_wgs.buffer(0)
        if not other_poly_wgs.is_valid: other_poly_wgs = other_poly_wgs.buffer(0)

        if not base_poly_wgs.is_valid or not other_poly_wgs.is_valid:
            return None

        if base_poly_wgs.intersects(other_poly_wgs):
            overlap_geom_wgs84 = base_poly_wgs.intersection(other_poly_wgs)
            if overlap_geom_wgs84.is_empty or not overlap_geom_wgs84.is_valid:
                overlap_geom_wgs84 = None
            elif not isinstance(overlap_geom_wgs84, (Polygon, gpd.GeoSeries, gpd.GeoDataFrame)):
                overlap_geom_wgs84 = None
        else:
            overlap_geom_wgs84 = None

    except Exception:
        overlap_geom_wgs84 = None
        return None

    # Calculate cloud cover in the overlap area
    cloud_overlap_base = 1.0
    cloud_overlap_other = 1.0
    better_img_in_overlap = None

    if overlap_geom_wgs84:
        cloud_overlap_base = get_cloud_cover_in_geom(base_img, overlap_geom_wgs84)
        cloud_overlap_other = get_cloud_cover_in_geom(other_img, overlap_geom_wgs84)
        better_img_in_overlap = base_img if cloud_overlap_base <= cloud_overlap_other else other_img

        overlap_details = {
            'overlap_geometry_wgs84': mapping(overlap_geom_wgs84),
            'cloud_cover_base_in_overlap': cloud_overlap_base,
            'cloud_cover_other_in_overlap': cloud_overlap_other,
            'prioritized_image_filename': better_img_in_overlap.get('filename') if better_img_in_overlap else None
        }
    else:
        overlap_details = {'overlap_exists': False}

    # Calculate quality factors
    quality_base = (1.0 - base_img.get('cloud_coverage', 1.0)) * base_img.get('valid_pixels_percentage', 0.0)
    quality_other = (1.0 - other_img.get('cloud_coverage', 1.0)) * other_img.get('valid_pixels_percentage', 0.0)

    # Calculate overlap quality
    if better_img_in_overlap:
        cloud_better_overlap = min(cloud_overlap_base, cloud_overlap_other)
        valid_pix_better_overlap = better_img_in_overlap.get('valid_pixels_percentage', 0.0)
        quality_factor_overlap = (1.0 - cloud_better_overlap) * valid_pix_better_overlap
    else:
        quality_factor_overlap = (quality_base + quality_other) / 2.0

    # Calculate refined quality factor
    refined_quality_factor = ((1.0 - OVERLAP_QUALITY_WEIGHT) * ((quality_base + quality_other) / 2.0) +
                             OVERLAP_QUALITY_WEIGHT * quality_factor_overlap)

    # Calculate estimated coverage
    uncovered_area_before = max(0.0, 1.0 - base_img.get('geographic_coverage', 0.0))
    overlap_factor_heuristic = 0.4 if other_img.get('class') == 'central' else 0.2
    contribution_factor = (1.0 - overlap_factor_heuristic)
    added_geo_coverage_est = min(uncovered_area_before, other_img.get('geographic_coverage', 0.0) * contribution_factor)
    estimated_new_geo_coverage = min(1.0, base_img.get('geographic_coverage', 0.0) + added_geo_coverage_est)

    # Calculate final effectiveness score
    effectiveness_score = (added_geo_coverage_est * refined_quality_factor) + orbit_bonus

    return {
        'image': other_img,
        'days_diff': days_diff,
        'estimated_coverage_after_add': estimated_new_geo_coverage,
        'refined_quality_factor': refined_quality_factor,
        'effectiveness_score': effectiveness_score,
        'orbit_match': orbit_match,
        'overlap_details': overlap_details
    }

def find_mosaic_combinations(image_metadata: dict, max_days_diff: int) -> list:
    logging.info("\nAnalyzing potential mosaic combinations...")
    potential_mosaics = []
    centrals = image_metadata.get('central', [])
    complements = image_metadata.get('complement', [])
    all_accepted = centrals + complements

    if not all_accepted:
        logging.warning("No accepted central or complement images found to create mosaics.")
        return []

    processed_centrals = set()
    for central_img in centrals:
        if central_img['filename'] in processed_centrals: continue

        compatible_images_data = []
        # Check compatibility with complements
        for comp_img in complements:
            comp_data = calculate_refined_compatibility(central_img, comp_img, max_days_diff)
            if comp_data:
                compatible_images_data.append(comp_data)

        # Check compatibility with other centrals
        for other_central in centrals:
            if other_central['filename'] == central_img['filename']: continue
            comp_data = calculate_refined_compatibility(central_img, other_central, max_days_diff)
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

            # Aggregate overlap details from the top additions
            combined_overlap_details = []
            for item in top_additions_data:
                if item.get('overlap_details'):
                    combined_overlap_details.append({
                        'base_image': central_img['filename'],
                        'other_image': item['image']['filename'],
                        **item['overlap_details']
                    })

            mosaic = {
                'type': 'mixed_mosaic',
                'base_image': central_img['filename'],
                'component_images': all_filenames,
                'estimated_coverage': estimated_final_coverage,
                'avg_quality_factor': avg_quality_factor,
                'time_window_start': min(all_dates).isoformat(),
                'time_window_end': max(all_dates).isoformat(),
                'component_details': [{'filename': img['filename'], 'class': img['class'], 'date': img['date'].isoformat() if isinstance(img.get('date'), datetime) else None} for img in mosaic_components],
                'contains_sar': True,
                'overlap_details': combined_overlap_details
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
                    'contains_sar': True,
                    'overlap_details': []
                }
                potential_mosaics.append(mosaic)

    logging.info(f"Identified {len(potential_mosaics)} potential mosaic combinations.")
    potential_mosaics.sort(key=lambda x: (x.get('estimated_coverage', 0.0), x.get('avg_quality_factor', 0.0)), reverse=True)
    return potential_mosaics

# --- Validation Functions ---

def find_worst_combinations_per_period(all_images_metadata: list, aoi_area: float, period_days: int = 15) -> list:
    """
    Encontra exemplos de combinações ruins (alta nuvem, baixa cobertura, rejeitados)
    em cada período de tempo. Retorna TODOS os exemplos encontrados que atendem aos critérios
    e cujos paths TCI são válidos.
    """
    logging.info(f"\nSearching for ALL bad mosaic examples per {period_days}-day period...")
    all_bad_combinations = [] # <<< Coletará todos os exemplos ruins
    images_by_period = defaultdict(list)
    valid_images_for_period = [img for img in all_images_metadata if isinstance(img.get('date'), datetime)]
    if not valid_images_for_period:
        logging.warning("No image metadata with dates provided to find worst combinations.")
        return []
    try: min_date = min(img['date'] for img in valid_images_for_period)
    except ValueError: logging.warning("Could not determine minimum date from images."); return []

    for img_meta in valid_images_for_period:
        delta_days = (img_meta['date'] - min_date).days
        period_index = delta_days // period_days
        images_by_period[period_index].append(img_meta)

    for period_index, images_in_period in sorted(images_by_period.items()):
        start_date = min_date + timedelta(days=period_index * period_days)
        end_date = start_date + timedelta(days=period_days - 1)
        logging.info(f" Analyzing period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} ({len(images_in_period)} images)")
        # Considera imagens com bounds, crs e um path TCI (final ou temporário)
        plottable_images = [img for img in images_in_period if img.get('bounds') and img.get('crs') and (img.get('tci_path') or img.get('temp_tci_path'))]
        if len(plottable_images) < 2:
            logging.info(f"  Skipping period, less than 2 plottable images found.")
            continue

        potential_bad_combinations_in_period = []
        # Critério 1: Par com maior nuvem (se > 50%)
        plottable_images.sort(key=lambda x: x.get('cloud_coverage', 0.0), reverse=True)
        if plottable_images[0].get('cloud_coverage', 0) > 0.5:
             potential_bad_combinations_in_period.append({'criteria': 'highest_cloud', 'images': plottable_images[:2]})

        # Critério 2: Par com menor cobertura geográfica (mas > 0)
        plottable_images.sort(key=lambda x: x.get('geographic_coverage', 0.0))
        lowest_geo_pair = [img for img in plottable_images if img.get('geographic_coverage', 0) > 0.0][:2]
        if len(lowest_geo_pair) == 2:
             potential_bad_combinations_in_period.append({'criteria': 'lowest_geo_coverage', 'images': lowest_geo_pair})

        # Critério 3: Par de imagens rejeitadas
        rejected_plottable = [img for img in plottable_images if img.get('status') == 'rejected']
        if len(rejected_plottable) >= 2:
             rejected_plottable.sort(key=lambda x: x.get('filename')) # Ordena para consistência
             potential_bad_combinations_in_period.append({'criteria': 'rejected_pair', 'images': rejected_plottable[:2]})
        # Critério 4: Uma rejeitada + a com pior cobertura geográfica
        elif len(rejected_plottable) == 1:
             other_plottable = [img for img in plottable_images if img['filename'] != rejected_plottable[0]['filename']]
             if other_plottable:
                 other_plottable.sort(key=lambda x: x.get('geographic_coverage', 0.0))
                 potential_bad_combinations_in_period.append({'criteria': 'single_rejected_plus_worst', 'images': [rejected_plottable[0], other_plottable[0]]})

        if not potential_bad_combinations_in_period:
             logging.info("  No candidate 'bad' pairs found meeting criteria for this period.")
             continue

        processed_pairs_in_period = set() # Evitar duplicatas exatas de pares
        for combo in potential_bad_combinations_in_period:
            if len(combo['images']) != 2: continue
            img1, img2 = combo['images']

            pair_key = tuple(sorted([img1['filename'], img2['filename']]))
            if pair_key in processed_pairs_in_period: continue
            processed_pairs_in_period.add(pair_key)

def _scale_uint16_data(data: np.ma.MaskedArray, filename: str) -> np.ndarray | None:
    logging.info(f"    Scaling uint16 data for {filename} using percentile stretch.")
    try:
        valid_data = data[~data.mask]
        if valid_data.size == 0:
            logging.warning("    No valid data to calculate percentiles, converting directly.")
            # Retorna array de zeros com 3 canais se a entrada tiver 3 canais
            shape_out = (data.shape[1], data.shape[2], 3) if len(data.shape) == 3 else (data.shape[0], data.shape[1])
            return np.zeros(shape_out, dtype=np.uint8)

        p2, p98 = np.percentile(valid_data, (2, 98))
        if p98 > p2:
            data_float = data.astype(np.float32)
            scaled_data = (data_float - p2) / (p98 - p2)
            scaled_data = np.clip(scaled_data, 0, 1)
            scaled_uint8 = (scaled_data * 255.0).astype(np.uint8)
            logging.info(f"    Scaled using p2={p2:.2f}, p98={p98:.2f}")
            return np.ma.filled(scaled_uint8, 0)
        elif data.max() > 0:
             logging.warning("    Percentiles equal, falling back to max scaling.")
             data_float = data.astype(np.float32)
             max_val = data_float.max()
             if max_val > 0:
                 scaled_uint8 = (data_float / max_val * 255.0).astype(np.uint8)
                 return np.ma.filled(scaled_uint8, 0)
             else:
                 logging.warning("    Data max is 0 after masking, converting directly.")
                 return np.ma.filled(data.astype(np.uint8), 0)
        else:
             logging.warning("    Data max is 0 or percentiles equal at 0, converting directly.")
             return np.ma.filled(data.astype(np.uint8), 0)
    except Exception as e_scale:
        logging.error(f"    Error during uint16 scaling: {e_scale}. Attempting direct conversion.")
        try:
            return np.ma.filled(data.astype(np.uint8), 0)
        except ValueError:
            logging.error(f"    Cannot convert uint16 data to uint8.")
            return None

# _plot_mosaic_raster: SEM MUDANÇAS (mas incluída para contexto)
def _plot_mosaic_raster(
    component_details: list,
    aoi_gdf: gpd.GeoDataFrame,
    plot_index: int,
    output_plot_dir: Path,
    plot_type: str,
    title: str,
    subtitle: str,
    filename_base: str
):
    logging.info(f"  Generating {plot_type} raster plot: {filename_base}")
    fig, ax = plt.subplots(figsize=(12, 12), dpi=300)
    ax.set_aspect('equal')
    target_crs = aoi_gdf.crs

    aoi_gdf.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1.5, zorder=10, label='Área de Interesse')
    legend_elements = [plt.Line2D([0], [0], color='black', lw=1.5, label='Área de Interesse')] # Usar Line2D para Patch
    component_labels_added = set()

    if plot_type == 'bad':
        # Usar Line2D para consistência, ou Patch se preferir preenchimento
        legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', label='Componente TCI Subótimo',
                           markerfacecolor='red', markersize=10, alpha=0.6))
        component_labels_added.add('bad_component')

    for comp_detail in component_details:
        filename = comp_detail.get('filename', 'Unknown Filename')
        tci_path_str = comp_detail.get('tci_path')
        bounds = comp_detail.get('bounds') # Pode ser dict ou BoundingBox
        crs = comp_detail.get('crs')

        if not tci_path_str: logging.warning(f"    TCI path not found for component {filename}. Skipping in plot."); continue
        tci_path = Path(tci_path_str)
        if not tci_path.exists(): logging.warning(f"    TCI file not found for component {filename} at {tci_path}. Skipping in plot."); continue
        if not bounds: logging.warning(f"    Bounds not found for component {filename}. Skipping in plot."); continue
        if not crs: logging.warning(f"    CRS not found for component {filename}. Skipping in plot."); continue

        logging.info(f"    Processing TCI for imshow: {tci_path.name}")
        try:
            with rasterio.open(tci_path) as src:
                raster_dtype = src.dtypes[0]
                logging.info(f"      Raster Info - Shape: {src.shape}, Dtype: {raster_dtype}, Count: {src.count}, CRS: {src.crs}")
                if src.count < 3: logging.warning(f"      Skipping plot: TCI file {tci_path.name} has only {src.count} bands."); continue

                src_bounds = src.bounds # Sempre BoundingBox aqui
                src_crs = src.crs

                data = src.read([1, 2, 3], masked=True)
                if data.mask.all(): logging.warning(f"      Skipping plot: TCI file {tci_path.name} contains only masked/nodata values."); continue

                img_display = None
                if raster_dtype == 'uint16':
                    scaled_data_np = _scale_uint16_data(data, tci_path.name)
                    if scaled_data_np is None: continue
                    img_display = np.moveaxis(scaled_data_np, 0, -1)
                elif raster_dtype == 'uint8':
                    img_display = np.moveaxis(data.filled(0), 0, -1)
                else:
                    logging.warning(f"      Unexpected data type {raster_dtype} for {tci_path.name}. Attempting conversion.")
                    if np.issubdtype(raster_dtype, np.floating):
                         data_uint8 = (np.clip(data.filled(0), 0, 1) * 255.0).astype(np.uint8)
                         img_display = np.moveaxis(data_uint8, 0, -1)
                    else:
                         try:
                             img_display = np.moveaxis(data.filled(0).astype(np.uint8), 0, -1)
                         except ValueError: logging.error(f"      Cannot convert {raster_dtype} data to uint8 for display."); continue

                if img_display is None or img_display.size == 0:
                    logging.warning(f"      Image data for display is None or empty for {tci_path.name}. Skipping imshow."); continue

                left, bottom, right, top = transform_bounds(src_crs, target_crs, *src_bounds)
                extent = [left, right, bottom, top]

                if plot_type == 'good':
                    img_class = comp_detail.get('class', 'unknown')
                    alpha = 0.7 if img_class == 'central' else 0.6
                    zorder = 1 if img_class == 'central' else 2
                    color_for_legend = 'blue' if img_class == 'central' else 'green' if img_class == 'complement' else 'gray'
                    label = f'Componente {img_class.capitalize()}'
                    if img_class not in component_labels_added:
                         # Usar Line2D para consistência
                         legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', label=label,
                                            markerfacecolor=color_for_legend, markersize=10, alpha=alpha))
                         component_labels_added.add(img_class)
                else: # plot_type == 'bad'
                    alpha = 0.6
                    zorder = 1
                    # A legenda para 'bad' já foi adicionada no início

                ax.imshow(img_display, extent=extent, alpha=alpha, zorder=zorder, interpolation='nearest')
                logging.info(f"      ax.imshow executed for {tci_path.name}")

                # Adicionar anotação no centro da imagem
                try:
                    center_src = ((src_bounds.left + src_bounds.right) / 2, (src_bounds.bottom + src_bounds.top) / 2)
                    xs, ys = transform(src_crs, target_crs, [center_src[0]], [center_src[1]])
                    center_plot = (xs[0], ys[0])

                    annotation_text = ""
                    if plot_type == 'good':
                        img_date_str = comp_detail.get('date', 'N/A')
                        if isinstance(img_date_str, str): img_date_str = img_date_str.split('T')[0]
                        annotation_text = f"{comp_detail.get('class', '').capitalize()}\n{img_date_str}"
                    else: # plot_type == 'bad'
                        status = comp_detail.get('status', 'unknown').capitalize()
                        reason = comp_detail.get('reason', '')
                        cloud_cov = comp_detail.get('cloud_cov', -1)
                        eff_cov = comp_detail.get('eff_cov', -1)
                        annotation_text = f"Status: {status}"
                        if 'nuvens' in reason: annotation_text += f"\nNuvens: {cloud_cov:.1%}"
                        elif 'EFETIVA' in reason: annotation_text += f"\nCob.Efetiva: {eff_cov:.1%}"
                        elif 'GEOGRÁFICA' in reason: annotation_text += f"\nCob.Geo: {comp_detail.get('geo_cov', -1):.1%}"

                    if annotation_text:
                         ax.annotate(annotation_text, xy=center_plot, xytext=(0, 0), textcoords='offset points',
                                     ha='center', va='center', fontsize=7, color='white',
                                     bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.6))
                except Exception as e_annot: logging.warning(f"    Could not add annotation for {filename}: {e_annot}")

        except rasterio.RasterioIOError as e_rio: logging.error(f"    Rasterio error opening {tci_path.name}: {e_rio}")
        except Exception as e_plot_raster: logging.error(f"    Failed during raster processing/plotting for {tci_path.name}: {e_plot_raster}\n{traceback.format_exc()}")

    # Ajustar limites do plot com base na AOI
    minx, miny, maxx, maxy = aoi_gdf.total_bounds
    width, height = maxx - minx, maxy - miny
    padding_x, padding_y = (width * 0.1 if width > 0 else 1), (height * 0.1 if height > 0 else 1)
    ax.set_xlim(minx - padding_x, maxx + padding_x)
    ax.set_ylim(miny - padding_y, maxy + padding_y)

    # Títulos e legenda
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.text(0.5, 1.02, subtitle, transform=ax.transAxes, ha='center', va='bottom', fontsize=11)
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    ax.grid(True, linestyle='--', color='gray', alpha=0.3)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout(rect=[0, 0, 1, 0.98]) # Ajusta para não cortar títulos/legendas

    # Salvar plot
    plot_filename = output_plot_dir / f"{filename_base}.png"
    try:
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        logging.info(f"  Raster plot saved to {plot_filename}")
    except Exception as e_save:
        logging.error(f"  Failed to save raster plot {plot_filename}: {e_save}")
    finally:
        plt.close(fig) # Fecha a figura para liberar memória

# Função modificada
def plot_mosaic_composition(
    mosaic_info: dict,
    metadata_lookup: dict | None,
    aoi_gdf: gpd.GeoDataFrame,
    plot_index: int, # Usado para garantir nomes únicos se datas forem iguais
    output_plot_dir: Path,
    plot_type: str
):
    component_details = []
    title = ""
    subtitle = ""
    filename_base = ""
    date_str = "NODATE" # Data para o nome do arquivo

    if plot_type == 'good':
        if not metadata_lookup:
            logging.error("Metadata lookup is required for plotting 'good' mosaics.")
            return

        mosaic_type_desc = mosaic_info.get('type', 'mosaic')
        logging.info(f"Preparing data for GOOD mosaic plot {plot_index + 1} ({mosaic_type_desc})...")
        component_filenames = mosaic_info.get('component_images', [])
        if not component_filenames:
             logging.warning(f"No component images found in 'good' mosaic info {plot_index + 1}. Skipping plot.")
             return

        # Extrair data para nome do arquivo
        try:
            start_date_iso = mosaic_info.get('time_window_start')
            if start_date_iso: date_str = start_date_iso.split('T')[0] # Pega YYYY-MM-DD
        except Exception: pass

        for filename in component_filenames:
            img_meta = metadata_lookup.get(filename)
            if not img_meta: logging.warning(f"Metadata not found for {filename} in lookup. Skipping in plot."); continue
            tci_path_str = img_meta.get('tci_path') or img_meta.get('temp_tci_path')
            if not tci_path_str: logging.warning(f"TCI path not found for {filename}. Skipping in plot."); continue

            details = {
                'filename': filename, 'tci_path': tci_path_str, 'bounds': img_meta.get('bounds'),
                'crs': img_meta.get('crs'), 'class': img_meta.get('class'), 'date': img_meta.get('date')
            }
            # Converter bounds dict para BoundingBox se necessário
            if isinstance(details['bounds'], dict) and all(k in details['bounds'] for k in ['left', 'bottom', 'right', 'top']):
                 try: details['bounds'] = rasterio.coords.BoundingBox(**details['bounds'])
                 except Exception as e_bounds: logging.warning(f"Could not convert bounds dict for {filename}: {e_bounds}"); details['bounds'] = None
            elif not isinstance(details.get('bounds'), rasterio.coords.BoundingBox):
                 logging.warning(f"Invalid bounds format for {filename}: {type(details.get('bounds'))}"); details['bounds'] = None

            component_details.append(details)

        if not component_details:
            logging.warning(f"No valid component details found for good mosaic {plot_index + 1}. Skipping plot.")
            return

        est_cov = mosaic_info.get('estimated_coverage', -1)
        avg_qual = mosaic_info.get('avg_quality_factor', -1)
        title = f"Composição de Mosaico Otimizada {plot_index + 1} ({mosaic_type_desc})"
        subtitle = f"Cob. Estimada: {est_cov:.1%} | Qualidade Média: {avg_qual:.2f} | Imagens: {len(component_filenames)}"
        # <<< NOVO NOME DO ARQUIVO COM DATA >>>
        filename_base = f"{date_str}_good_mosaic_{plot_index + 1}_{mosaic_type_desc}_raster"

    elif plot_type == 'bad':
        criteria_used = mosaic_info.get('criteria_used', 'unknown')
        logging.info(f"Preparing data for BAD mosaic plot {plot_index + 1} (Criteria: {criteria_used})...")
        component_details_raw = mosaic_info.get('component_details', []) # Pega os detalhes brutos
        if not component_details_raw:
            logging.warning(f"No component details found for bad mosaic {plot_index + 1}. Skipping plot.")
            return

        # Extrair data para nome do arquivo
        try:
            period_start_str = mosaic_info.get('period_start')
            if period_start_str: date_str = period_start_str # Usa YYYY-MM-DD do período
        except Exception: pass

        # Processar detalhes, convertendo bounds
        for detail in component_details_raw:
            # Converter bounds dict para BoundingBox se necessário
            if isinstance(detail.get('bounds'), dict) and all(k in detail['bounds'] for k in ['left', 'bottom', 'right', 'top']):
                 try: detail['bounds'] = rasterio.coords.BoundingBox(**detail['bounds'])
                 except Exception as e_bounds: logging.warning(f"Could not convert bounds dict for {detail.get('filename')}: {e_bounds}"); detail['bounds'] = None
            elif not isinstance(detail.get('bounds'), rasterio.coords.BoundingBox):
                 logging.warning(f"Invalid bounds format for {detail.get('filename')}: {type(detail.get('bounds'))}"); detail['bounds'] = None
            component_details.append(detail) # Adiciona o detalhe processado

        criteria_map = {'highest_cloud': 'NuvensAltas', 'lowest_geo_coverage': 'CobGeoBaixa', 'rejected_pair': 'ParRejeitado', 'single_rejected_plus_worst': 'RejeitadoPiorCob'}
        criteria_desc_short = criteria_map.get(criteria_used, criteria_used)
        title = f"Exemplo de Mosaico Subótimo {plot_index + 1} (Critério: {criteria_desc_short})"
        period_start = mosaic_info.get('period_start', 'N/A')
        period_end = mosaic_info.get('period_end', 'N/A')
        est_eff_cov = mosaic_info.get('estimated_combined_effective', -1.0)
        subtitle = f"Período: {period_start} a {period_end} | Cob. Efetiva Estimada: {est_eff_cov:.1%}"
        # <<< NOVO NOME DO ARQUIVO COM DATA E CRITÉRIO >>>
        filename_base = f"{date_str}_bad_example_{plot_index + 1}_{criteria_desc_short}_raster"

    else:
        logging.error(f"Invalid plot_type '{plot_type}' provided to plot_mosaic_composition.")
        return

    # Filtrar componentes sem bounds válidos antes de plotar
    valid_component_details = [d for d in component_details if isinstance(d.get('bounds'), rasterio.coords.BoundingBox)]
    if not valid_component_details:
        logging.warning(f"No components with valid bounds found for plot {filename_base}. Skipping plot.")
        return

    _plot_mosaic_raster(
        component_details=valid_component_details, # <<< Passa apenas detalhes válidos
        aoi_gdf=aoi_gdf,
        plot_index=plot_index,
        output_plot_dir=output_plot_dir,
        plot_type=plot_type,
        title=title,
        subtitle=subtitle,
        filename_base=filename_base
    )

# --- Main Pipeline ---

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime): return obj.isoformat()
        if isinstance(obj, rasterio.coords.BoundingBox): return {'left': obj.left, 'bottom': obj.bottom, 'right': obj.right, 'top': obj.top}
        return json.JSONEncoder.default(self, obj)

# Função modificada
def run_processing_pipeline():
    start_time = datetime.now()
    logging.info(f"Pipeline started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
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
                else: stats['accepted_unknown_class'] += 1; logging.warning(f"Accepted image {result.get('filename')} has unknown class: {img_class}")
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
                else: stats['rejected_unknown'] += 1; logging.warning(f"Unknown rejection reason: {reason}")
            elif status == 'error':
                stats['error'] += 1
                if 'Coverage/Cloud calculation error' in reason: stats['error_coverage_cloud'] += 1
                elif 'CRS' in reason or 'transform' in reason: stats['error_crs'] += 1
                elif 'Critical error' in reason or 'returned None' in reason: stats['error_critical'] += 1
                else: stats['error_unknown'] += 1; logging.warning(f"Unknown error reason: {reason}")
            else: stats['error_unknown_status'] += 1; logging.error(f"Unknown status code '{status}' for {result.get('filename')}")
        else:
            logging.error(f"Processing function returned None for {zip_path.name}. Recording as critical error.")
            result = {'status': 'error', 'reason': 'Processing function returned None', 'filename': zip_path.name}
            all_processed_metadata.append(result)
            stats['error'] += 1
            stats['error_critical'] += 1

    all_metadata_path = METADATA_DIR / 'all_processed_images_log.json'
    try:
        with open(all_metadata_path, 'w') as f: json.dump(all_processed_metadata, f, indent=2, cls=DateTimeEncoder)
        logging.info(f"Full processing log saved to: {all_metadata_path}")
    except IOError as e: logging.error(f"Failed to save full processing log: {e}")
    except TypeError as e_serial: logging.error(f"Serialization error saving full processing log: {e_serial}")

    metadata_lookup = {img['filename']: img for img in all_processed_metadata if img.get('filename')}
    logging.info("\n--- Finding GOOD Mosaic Combinations ---")
    logging.info(f"Central images available: {len(image_metadata.get('central', []))}")
    logging.info(f"Complement images available: {len(image_metadata.get('complement', []))}")
    good_mosaic_combinations = find_mosaic_combinations(image_metadata, MOSAIC_TIME_WINDOW_DAYS)
    logging.info(f"Found {len(good_mosaic_combinations)} potential GOOD mosaic combinations.")

    # --- Criação do optimization_params.json ---
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
         # Adiciona avg_cloud_coverage se existir no mosaic dict
         avg_cloud_cov = mosaic.get('avg_cloud_coverage', 1.0) # Pega do dict ou default
         optimization_params['mosaic_groups'].append({
             'group_id': f'mosaic_{idx+1}', 'type': mosaic['type'], 'images': mosaic['component_images'],
             'estimated_coverage': mosaic['estimated_coverage'], 'quality_factor': mosaic.get('avg_quality_factor', 0.0),
             'avg_cloud_coverage': avg_cloud_cov, # <<< Inclui a média de nuvens
             'time_window_start': mosaic['time_window_start'], 'time_window_end': mosaic['time_window_end'],
             'contains_sar': mosaic.get('contains_sar', False),
             'overlap_details': mosaic.get('overlap_details', [])
         })
    opt_params_path = METADATA_DIR / 'optimization_parameters.json'
    try:
        with open(opt_params_path, 'w') as f: json.dump(optimization_params, f, indent=2)
        logging.info(f"Optimization parameters saved to: {opt_params_path}")
    except IOError as e: logging.error(f"Failed to save optimization parameters: {e}")
    except TypeError as e_serial: logging.error(f"Serialization error saving optimization parameters: {e_serial}")

    # --- Generating plots for ALL GOOD mosaic compositions --- <<< MUDANÇA AQUI
    num_good_mosaics_to_plot = len(good_mosaic_combinations) # <<< REMOVIDO O LIMITE DE 10
    logging.info(f"\n--- Generating plots for ALL {num_good_mosaics_to_plot} GOOD mosaic compositions ---")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    if not good_mosaic_combinations: logging.warning("No good mosaic combinations found to plot.")
    for idx, mosaic in enumerate(good_mosaic_combinations): # <<< Loop sobre TODOS
        logging.info(f"Plotting good mosaic {idx + 1}/{num_good_mosaics_to_plot}...")
        try:
            plot_mosaic_composition(
                mosaic_info=mosaic,
                metadata_lookup=metadata_lookup,
                aoi_gdf=aoi_gdf_union_wgs84,
                plot_index=idx,
                output_plot_dir=PLOTS_DIR, # <<< Salva em publication_plots
                plot_type='good'
            )
        except Exception as e_plot:
            logging.error(f"  Failed to plot good mosaic {idx + 1}: {e_plot}\n{traceback.format_exc()}")
            stats['good_mosaic_plot_error'] += 1 # Contar erros de plotagem
    logging.info(f"Finished attempting to plot good mosaics. Plots saved in: {PLOTS_DIR}")

    # --- Generating plots for ALL BAD mosaic examples --- <<< MUDANÇA AQUI
    all_bad_mosaic_examples = find_worst_combinations_per_period(all_processed_metadata, aoi_area_wgs84, period_days=15) # <<< Chama a função modificada

    # Salvar metadados dos exemplos ruins na pasta TRASH
    bad_examples_meta_path = TRASH_DIR / 'all_bad_examples_metadata.json'
    try:
        with open(bad_examples_meta_path, 'w') as f: json.dump(all_bad_mosaic_examples, f, indent=2, cls=DateTimeEncoder)
        logging.info(f"All bad mosaic examples metadata saved to: {bad_examples_meta_path}")
    except Exception as e: logging.error(f"Failed to save bad examples metadata: {e}")

    logging.info(f"\nGenerating plots for {len(all_bad_mosaic_examples)} BAD mosaic examples...")
    TRASH_DIR.mkdir(parents=True, exist_ok=True) # Garante que a pasta trash existe
    for idx, bad_mosaic in enumerate(all_bad_mosaic_examples): # <<< Loop sobre TODOS os exemplos ruins
         logging.info(f"Plotting bad mosaic example {idx + 1}/{len(all_bad_mosaic_examples)}...")
         try:
             plot_mosaic_composition(
                 mosaic_info=bad_mosaic,
                 metadata_lookup=None, # Não necessário para 'bad'
                 aoi_gdf=aoi_gdf_union_wgs84,
                 plot_index=idx,
                 output_plot_dir=TRASH_DIR, # <<< Salva na pasta TRASH
                 plot_type='bad'
             )
         except Exception as e_plot:
             logging.error(f"  Failed to plot bad mosaic example {idx + 1}: {e_plot}\n{traceback.format_exc()}")
             stats['bad_example_plot_error'] += 1 # Contar erros de plotagem
    logging.info(f"Bad mosaic example plots saved in: {TRASH_DIR}")

    # --- Final Statistics and Summary ---
    print("\n" + "="*60 + "\nFINAL PROCESSING STATISTICS:\n" + "="*60)
    total_processed, total_accepted, total_rejected, total_errors = stats['processed'], stats['accepted'], stats['rejected'], stats['error']
    calculated_total = total_accepted + total_rejected + total_errors
    if calculated_total != total_processed and total_processed > 0: logging.warning(f"Statistics check mismatch: Accepted({total_accepted}) + Rejected({total_rejected}) + Errors({total_errors}) = {calculated_total} != Processed({total_processed}).")
    print(f"Total ZIP files found:          {total_zip_files}")
    print(f"Total ZIP files processed:      {total_processed}")
    print("-" * 30)
    print(f"Total images accepted:          {total_accepted} ({total_accepted/total_processed*100 if total_processed else 0:.1f}%)")
    print(f"  - Accepted (files copied OK): {total_accepted - stats['accepted_copy_error']}")
    print(f"  - Accepted (file copy error): {stats['accepted_copy_error']}")
    print(f"    - Central images:           {stats['central']}")
    print(f"    - Complement images:        {stats['complement']}")
    if stats['accepted_unknown_class'] > 0: print(f"    - Unknown Class:            {stats['accepted_unknown_class']}")
    print("-" * 30)
    print(f"Total images rejected:          {total_rejected} ({total_rejected/total_processed*100 if total_processed else 0:.1f}%)")
    print(f"  - Rejection Reasons:")
    print(f"    - Extraction/Bad Zip:       {stats['rejected_extraction']}")
    print(f"    - Missing Files:            {stats['rejected_missing_files']}")
    print(f"    - Date Extraction Failed:   {stats['rejected_date']}")
    print(f"    - Sem Pixels Válidos na AOI:{stats['rejected_no_valid_pixels']}")
    print(f"    - Insufficient Geo Cov:     {stats['rejected_geo_coverage']}")
    print(f"    - Insufficient Eff Cov:     {stats['rejected_eff_coverage']}")
    print(f"    - High Cloud Coverage:      {stats['rejected_cloud']}")
    if stats['rejected_unknown'] > 0: print(f"    - Unknown Reason:           {stats['rejected_unknown']}")
    print("-" * 30)
    print(f"Total processing errors:        {total_errors} ({total_errors/total_processed*100 if total_processed else 0:.1f}%)")
    print(f"  - Error Reasons:")
    print(f"    - Coverage/Cloud Calc:      {stats['error_coverage_cloud']}")
    print(f"    - CRS/Transform Error:      {stats['error_crs']}")
    print(f"    - Critical/Other Error:     {stats['error_critical']}")
    if stats['error_unknown'] > 0: print(f"    - Unknown Error:            {stats['error_unknown']}")
    if stats['error_unknown_status'] > 0: print(f"    - Unknown Status Code:      {stats['error_unknown_status']}")
    if stats['good_mosaic_plot_error'] > 0: print(f"Errors plotting good mosaics:   {stats['good_mosaic_plot_error']}")
    if stats['bad_example_plot_error'] > 0: print(f"Errors plotting bad examples:   {stats['bad_example_plot_error']}")
    print("="*60)
    print(f"Potential GOOD mosaic combinations identified: {len(good_mosaic_combinations)}")
    print(f"Examples of BAD mosaic combinations identified: {len(all_bad_mosaic_examples)}")
    print("-" * 30)
    print(f"Metadata saved in:              {METADATA_DIR}")
    print(f"  - Full processing log:        {all_metadata_path.name}")
    print(f"  - Optimization parameters:    {opt_params_path.name}")
    print(f"GOOD Mosaic plots saved in:     {PLOTS_DIR}")
    print(f"BAD Example plots saved in:     {TRASH_DIR}") # <<< Atualizar caminho
    print(f"  - Bad examples metadata:    {bad_examples_meta_path.name} (in {TRASH_DIR})") # <<< Atualizar caminho
    print(f"Rejected TCIs copied to:        {VALIDATION_TCIS_DIR}")
    print("="*60)
    end_time = datetime.now()
    logging.info(f"Pipeline finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Total execution time: {end_time - start_time}")
    print("\nProcessing pipeline finished!")

# --- Script Execution ---
if __name__ == "__main__":
    run_processing_pipeline()