import zipfile
from pathlib import Path
import logging
import geopandas as gpd
from datetime import datetime, timedelta
import shutil
import json
import traceback
import numpy as np
import rasterio
from rasterio.warp import transform_bounds
from collections import defaultdict
from shapely.geometry import box, Polygon
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pyproj
from shapely.ops import transform as shapely_transform
from rasterio import features

from greedy_utils.configuration import *
from greedy_utils.file_utils import safe_extract, remove_dir_contents
from greedy_utils.metadata_utils import get_date_from_xml, extract_orbit_from_filename, save_classification_metadata, get_cloud_cover_in_geom
from greedy_utils.image_processing import calculate_coverage_metrics, calculate_cloud_coverage, check_image_suitability, classify_image # Remover calculate_c se não for usado
from greedy_utils.plotting_utils import find_worst_combinations_per_period, plot_mosaic_composition

# Configure matplotlib for plots
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'Arial'

# Use the same logging configuration as the original
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime): return obj.isoformat()
        if isinstance(obj, rasterio.coords.BoundingBox): return {'left': obj.left, 'bottom': obj.bottom, 'right': obj.right, 'top': obj.top}
        return json.JSONEncoder.default(self, obj)

def process_zip_file(zip_path: Path, aoi_gdf_wgs84: gpd.GeoDataFrame, aoi_area_wgs84: float, stats: dict, index: int = 0,total: int = 0):
    """Processes a single zip file, extracting necessary files, calculating metrics, and returning metadata."""
    if total > 0:
        logging.info(f"Processing [{index+1}/{total}]: {zip_path.name}")
    else:
        logging.info(f"Processing: {zip_path.name}")
    temp_dir = TEMP_EXTRACT_DIR / zip_path.stem
    temp_dir.mkdir(exist_ok=True)
    
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
            stats['error_extract'] += 1
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
            stats['error_missing_files'] += 1
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
            stats['error_missing_date_xml'] += 1
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
             stats['error_crs'] += 1
             return result_data
        except Exception as e_cov:
            reason = f'Coverage/Cloud calculation error: {e_cov}'
            logging.error(f"  {reason} for {zip_path.name}. Skipping.\n{traceback.format_exc()}")
            result_data.update({'status': 'error', 'reason': reason, 'cloud_coverage': cloud_percentage})
            save_classification_metadata(METADATA_DIR, None, result_data, date_obj, orbit, zip_path.name)
            stats['error_coverage_cloud'] += 1
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

            if "SEM PIXELS VÁLIDOS" in reason:
                stats['rejected_no_valid_pixels'] += 1
            elif "GEOGRÁFICA INSUFICIENTE" in reason:
                stats['rejected_geo_coverage'] += 1
            elif "EFETIVA INSUFICIENTE" in reason:
                stats['rejected_eff_coverage'] += 1
            elif "Muitas nuvens" in reason:
                stats['rejected_cloud'] += 1
            else:
                stats['rejected_unknown'] += 1
            
            save_classification_metadata(METADATA_DIR, None, result_data, date_obj, orbit, zip_path.name)
            stats['rejected'] += 1
            return result_data

        image_class = classify_image(result_data['effective_coverage'])
        logging.info(f"  CLASSIFIED AS: {image_class.upper()}")
        result_data['class'] = image_class
        result_data['status'] = f'accepted_{image_class}'

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

        try:
            shutil.copy2(xml_path, final_xml_path)
            shutil.copy2(cloud_mask_path_temp, final_cloud_mask_path)
            shutil.copy2(rgb_image_path_temp, final_tci_path)
            logging.info(f"  Files copied to {output_dir}")
            
            if image_class == 'central':
                stats['accepted_central'] += 1
            else:
                stats['accepted_complement'] += 1
        except Exception as e_copy:
            logging.error(f"  Error copying files for {zip_path.name} to {output_dir}: {e_copy}")
            result_data['reason'] = 'Accepted but failed to copy files'
            result_data['status'] = 'accepted_copy_error'
            save_classification_metadata(METADATA_DIR, image_class, result_data, date_obj, orbit, zip_path.name)
            stats['accepted_copy_error'] += 1

        return result_data

    except Exception as e:
        logging.error(f"  Critical error processing {zip_path.name}: {e}\n{traceback.format_exc()}")
        result_data['reason'] = f'Critical error: {e}'
        result_data['status'] = 'error'
        save_classification_metadata(METADATA_DIR, None, result_data, result_data.get('date'), result_data.get('orbit'), zip_path.name)
        stats['error_critical'] += 1
        return result_data
    finally:
        if temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
            except Exception as e_clean:
                logging.error(f"Failed to remove temp directory tree {temp_dir}: {e_clean}")

def calculate_refined_compatibility(base_img: dict, other_img: dict, max_days: int) -> dict | None:
    """Diretamente do original: calcula a compatibilidade entre duas imagens para mosaicos"""
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
    
    # Verifica se bounds são válidos
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
            
    # Continua com a validação de CRS
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
            elif not isinstance(overlap_geom_wgs84, (Polygon)):
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

        try:
            from shapely.geometry import mapping
            overlap_details = {
                'overlap_geometry_wgs84': mapping(overlap_geom_wgs84),
                'cloud_cover_base_in_overlap': cloud_overlap_base,
                'cloud_cover_other_in_overlap': cloud_overlap_other,
                'prioritized_image_filename': better_img_in_overlap.get('filename') if better_img_in_overlap else None
            }
        except ImportError:
            overlap_details = {
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
    """Diretamente do original: encontra possíveis combinações de mosaicos"""
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

def main():
    """Main function to orchestrate the processing of zip files and mosaic combination analysis."""
    start_time = datetime.now()
    logging.info(f"Pipeline started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    stats = defaultdict(int)
    image_metadata = {'central': [], 'complement': []}
    all_processed_metadata = []

    # Criar diretórios necessários
    OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    VALIDATION_TCIS_DIR.mkdir(parents=True, exist_ok=True)
    TRASH_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

    # Limpar diretório temporário
    remove_dir_contents(TEMP_EXTRACT_DIR)

    try:
        aoi_gdf_wgs84 = gpd.read_file(AOI_SHAPEFILE)
        if not aoi_gdf_wgs84.crs or aoi_gdf_wgs84.crs.to_epsg() != 4326:
             logging.warning(f"AOI CRS is not EPSG:4326 ({aoi_gdf_wgs84.crs}). Reprojecting...")
             aoi_gdf_wgs84 = aoi_gdf_wgs84.to_crs("EPSG:4326")
        aoi_geometry_union_wgs84 = aoi_gdf_wgs84.union_all()
        aoi_area_wgs84 = aoi_geometry_union_wgs84.area # <<< Área WGS84 necessária para plots ruins
        aoi_gdf_union_wgs84 = gpd.GeoDataFrame(geometry=[aoi_geometry_union_wgs84], crs="EPSG:4326")
        logging.info(f"AOI loaded from {AOI_SHAPEFILE}. CRS: {aoi_gdf_union_wgs84.crs}. Area (WGS84): {aoi_area_wgs84:.6f} sq. degrees.")
    except Exception as e:
        logging.error(f"CRITICAL: Failed to load or process AOI file {AOI_SHAPEFILE}: {e}")
        return

    # Processar arquivos ZIP
    zip_files = sorted(list(ZIP_SOURCE_DIR.glob('S2*.zip')))
    total_zip_files = len(zip_files)
    logging.info(f"Found {total_zip_files} ZIP files matching 'S2*.zip' in {ZIP_SOURCE_DIR}")

    if total_zip_files == 0:
        logging.warning("No ZIP files found to process.")
        return

    for index, zip_path in enumerate(zip_files):
        stats['processed'] += 1
        result = process_zip_file(zip_path, aoi_gdf_union_wgs84, aoi_area_wgs84, stats, index, total_zip_files)

        if result:
            # Converter data para datetime se for string ISO
            if isinstance(result.get('date'), str):
                try: result['date'] = datetime.fromisoformat(result['date'])
                except (ValueError, TypeError): pass # Deixa como está se falhar

            all_processed_metadata.append(result)
            if result.get('status', '').startswith('accepted'):
                stats['accepted'] += 1 # Contador geral de aceitos
                if result.get('class') == 'central':
                    image_metadata['central'].append(result)
                    # stats['accepted_central'] já é incrementado em process_zip_file
                elif result.get('class') == 'complement':
                    image_metadata['complement'].append(result)
                    # stats['accepted_complement'] já é incrementado em process_zip_file
                else:
                    stats['accepted_unknown_class'] += 1
                    logging.warning(f"Accepted image {result.get('filename')} has unknown class: {result.get('class')}")
                if result.get('status') == 'accepted_copy_error':
                    stats['accepted_copy_error'] += 1 # Já contado em process_zip_file, mas pode ser útil aqui
            elif result.get('status') == 'rejected':
                # stats['rejected'] já é incrementado em process_zip_file
                pass # Contadores específicos de rejeição já estão em process_zip_file
            elif result.get('status') == 'error':
                # stats['error_critical'] ou outros já são incrementados em process_zip_file
                pass
            else:
                stats['error_unknown_status'] += 1
                logging.error(f"Unknown status code '{result.get('status')}' for {result.get('filename')}")
        else:
            logging.error(f"Processing function returned None for {zip_path.name}. Recording as critical error.")
            result_data = {'status': 'error', 'reason': 'Processing function returned None', 'filename': zip_path.name}
            all_processed_metadata.append(result_data)
            stats['error_critical'] += 1 # Incrementa erro crítico aqui também

    # Salvar metadados processados
    all_metadata_path = METADATA_DIR / 'all_processed_images_log.json'
    try:
        with open(all_metadata_path, 'w') as f:
            json.dump(all_processed_metadata, f, indent=2, cls=DateTimeEncoder)
        logging.info(f"Full processing log saved to: {all_metadata_path}")
    except Exception as e:
        logging.error(f"Failed to save full processing log: {e}")

    # <<< Criar metadata_lookup APÓS processar todos os arquivos >>>
    metadata_lookup = {img['filename']: img for img in all_processed_metadata if img.get('filename')}

    # Encontrar possíveis combinações de mosaico
    logging.info("\n--- Finding GOOD Mosaic Combinations ---")
    logging.info(f"Central images available: {len(image_metadata.get('central', []))}")
    logging.info(f"Complement images available: {len(image_metadata.get('complement', []))}")
    start_greedy_time = datetime.now()
    good_mosaic_combinations = find_mosaic_combinations(image_metadata, MOSAIC_TIME_WINDOW_DAYS)
    end_greedy_time = datetime.now()
    greedy_duration = end_greedy_time - start_greedy_time
    logging.info(f"Greedy Heuristic finished in: {greedy_duration}")
    logging.info(f"Found {len(good_mosaic_combinations)} potential GOOD mosaic combinations.")

    # Criar parâmetros de otimização
    optimization_params = {'image_catalog': [], 'mosaic_groups': []}

    for img_class in ['central', 'complement']:
        for img_meta in image_metadata.get(img_class, []):
            optimization_params['image_catalog'].append({
                'filename': img_meta['filename'],
                'class': img_class,
                'date': img_meta['date'].isoformat() if isinstance(img_meta.get('date'), datetime) else None,
                'orbit': img_meta['orbit'],
                'geographic_coverage': img_meta['geographic_coverage'],
                'effective_coverage': img_meta['effective_coverage'],
                'cloud_coverage': img_meta['cloud_coverage'],
                'quality_factor': (1.0 - img_meta.get('cloud_coverage', 1.0)) * img_meta.get('valid_pixels_percentage', 0.0)
            })

    for idx, mosaic in enumerate(good_mosaic_combinations):
        avg_cloud_cov = mosaic.get('avg_cloud_coverage', 1.0)
        optimization_params['mosaic_groups'].append({
            'group_id': f'mosaic_{idx+1}',
            'type': mosaic['type'],
            'images': mosaic['component_images'],
            'estimated_coverage': mosaic['estimated_coverage'],
            'quality_factor': mosaic.get('avg_quality_factor', 0.0),
            'avg_cloud_coverage': avg_cloud_cov,
            'time_window_start': mosaic['time_window_start'],
            'time_window_end': mosaic['time_window_end'],
            'contains_sar': mosaic.get('contains_sar', False),
            'overlap_details': mosaic.get('overlap_details', [])
        })

    opt_params_path = METADATA_DIR / 'optimization_parameters.json'
    try:
        with open(opt_params_path, 'w') as f:
            json.dump(optimization_params, f, indent=2)
        logging.info(f"Optimization parameters saved to: {opt_params_path}")
    except Exception as e:
        logging.error(f"Failed to save optimization parameters: {e}")

    # --- <<< INÍCIO: Geração de Plots (código do original) >>> ---
    num_good_mosaics_to_plot = len(good_mosaic_combinations)
    logging.info(f"\n--- Generating plots for ALL {num_good_mosaics_to_plot} GOOD mosaic compositions ---")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    if not good_mosaic_combinations: logging.warning("No good mosaic combinations found to plot.")
    for idx, mosaic in enumerate(good_mosaic_combinations):
        logging.info(f"Plotting good mosaic {idx + 1}/{num_good_mosaics_to_plot}...")
        try:
            plot_mosaic_composition(
                mosaic_info=mosaic,
                metadata_lookup=metadata_lookup, # <<< Passa o lookup criado
                aoi_gdf=aoi_gdf_union_wgs84,
                plot_index=idx,
                output_plot_dir=PLOTS_DIR, # <<< Usa PLOTS_DIR da config
                plot_type='good'
            )
        except Exception as e_plot:
            logging.error(f"  Failed to plot good mosaic {idx + 1}: {e_plot}\n{traceback.format_exc()}")
            stats['good_mosaic_plot_error'] += 1
    logging.info(f"Finished attempting to plot good mosaics. Plots saved in: {PLOTS_DIR}")

    # --- Generating plots for ALL BAD mosaic examples ---
    all_bad_mosaic_examples = find_worst_combinations_per_period(all_processed_metadata, aoi_area_wgs84, period_days=15)

    # Salvar metadados dos exemplos ruins na pasta TRASH
    bad_examples_meta_path = TRASH_DIR / 'all_bad_examples_metadata.json'
    try:
        with open(bad_examples_meta_path, 'w') as f: json.dump(all_bad_mosaic_examples, f, indent=2, cls=DateTimeEncoder)
        logging.info(f"All bad mosaic examples metadata saved to: {bad_examples_meta_path}")
    except Exception as e: logging.error(f"Failed to save bad examples metadata: {e}")

    logging.info(f"\nGenerating plots for {len(all_bad_mosaic_examples)} BAD mosaic examples...")
    TRASH_DIR.mkdir(parents=True, exist_ok=True)
    for idx, bad_mosaic in enumerate(all_bad_mosaic_examples):
         logging.info(f"Plotting bad mosaic example {idx + 1}/{len(all_bad_mosaic_examples)}...")
         try:
             plot_mosaic_composition(
                 mosaic_info=bad_mosaic,
                 metadata_lookup=None, # Não necessário para 'bad'
                 aoi_gdf=aoi_gdf_union_wgs84,
                 plot_index=idx,
                 output_plot_dir=TRASH_DIR, # <<< Usa TRASH_DIR da config
                 plot_type='bad'
             )
         except Exception as e_plot:
             logging.error(f"  Failed to plot bad mosaic example {idx + 1}: {e_plot}\n{traceback.format_exc()}")
             stats['bad_example_plot_error'] += 1
    logging.info(f"Bad mosaic example plots saved in: {TRASH_DIR}")
    # --- <<< FIM: Geração de Plots >>> ---

    # Sumário final
    print("\n" + "="*60 + "\nFINAL PROCESSING STATISTICS:\n" + "="*60)
    total_processed = stats['processed']
    # Recalcula totais aceitos/rejeitados/erros a partir dos contadores específicos para garantir consistência
    total_accepted = stats['accepted_central'] + stats['accepted_complement'] + stats['accepted_copy_error'] # Soma todos os aceitos
    total_rejected = stats['rejected_no_valid_pixels'] + stats['rejected_geo_coverage'] + \
                     stats['rejected_eff_coverage'] + stats['rejected_cloud'] + stats['rejected_unknown'] + \
                     stats['error_extract'] + stats['error_missing_files'] + stats['error_missing_date_xml'] # Soma rejeitados + erros de pré-processamento
    total_errors = stats['error_crs'] + stats['error_coverage_cloud'] + stats['error_critical'] + stats['error_unknown'] + stats['error_unknown_status'] # Soma erros de processamento

    calculated_total = total_accepted + total_rejected + total_errors
    if calculated_total != total_processed and total_processed > 0:
        logging.warning(f"Statistics check mismatch: Accepted({total_accepted}) + Rejected({total_rejected}) + Errors({total_errors}) = {calculated_total} != Processed({total_processed}). Review counters.")

    print(f"Total ZIP files found:          {total_zip_files}")
    print(f"Total ZIP files processed:      {total_processed}")
    print("-" * 30)
    print(f"Total images accepted:          {total_accepted} ({total_accepted/total_processed*100 if total_processed else 0:.1f}%)")
    print(f"  - Central images:             {stats['accepted_central']}")
    print(f"  - Complement images:          {stats['accepted_complement']}")
    print(f"  - Accepted with copy errors:  {stats['accepted_copy_error']}")
    if stats['accepted_unknown_class'] > 0: print(f"  - Unknown Class:            {stats['accepted_unknown_class']}")
    print("-" * 30)
    print(f"Total images rejected:          {total_rejected} ({total_rejected/total_processed*100 if total_processed else 0:.1f}%)")
    print(f"  - Rejection/Pre-proc Reasons:")
    print(f"    - Extraction/Bad Zip:       {stats['error_extract']}")
    print(f"    - Missing Files:            {stats['error_missing_files']}")
    print(f"    - Missing XML Date:         {stats['error_missing_date_xml']}")
    print(f"    - No Valid Pixels:          {stats['rejected_no_valid_pixels']}")
    print(f"    - Insufficient Geo Coverage:{stats['rejected_geo_coverage']}")
    print(f"    - Insufficient Eff Coverage:{stats['rejected_eff_coverage']}")
    print(f"    - High Cloud Coverage:      {stats['rejected_cloud']}")
    print(f"    - Unknown Reason:           {stats['rejected_unknown']}")
    print("-" * 30)
    print(f"Total processing errors:        {total_errors} ({total_errors/total_processed*100 if total_processed else 0:.1f}%)")
    print(f"  - Processing Error Reasons:")
    print(f"    - CRS/Transform Error:      {stats['error_crs']}")
    print(f"    - Coverage/Cloud Calc:      {stats['error_coverage_cloud']}")
    print(f"    - Critical Error:           {stats['error_critical']}")
    print(f"    - Unknown Error:            {stats['error_unknown']}")
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
    print(f"BAD Example plots saved in:     {TRASH_DIR}")
    print(f"  - Bad examples metadata:    {bad_examples_meta_path.name} (in {TRASH_DIR})")
    print(f"Rejected TCIs copied to:        {VALIDATION_TCIS_DIR}") # <<< Adicionado caminho dos TCIs rejeitados
    print("="*60)

    end_time = datetime.now()
    logging.info(f"Pipeline finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Total execution time: {end_time - start_time}")
    print(f"Greedy Heuristic Execution Time: {greedy_duration}")


if __name__ == "__main__":
    main()