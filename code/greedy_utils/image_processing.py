import rasterio
import rasterio.features as features
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, box
from .metadata_utils import get_cloud_cover_in_geom
import pyproj
from shapely.ops import transform as shapely_transform
import logging
import traceback

def calculate_coverage_metrics(raster_path, aoi_gdf_crs_raster, aoi_area_wgs84):
    metrics = {'geographic_coverage': 0.0, 'valid_pixels_percentage': 0.0, 'effective_coverage': 0.0, 
               'bounds': None, 'crs': None}
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
                mask = features.geometry_mask([aoi_geometry_transformed], out_shape=src.shape, transform=src.transform, 
                                           invert=True, all_touched=True)
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
                logging.warning(f"Masking error for {raster_path} with AOI: {e_mask}. Likely AOI outside raster bounds. "
                                f"Valid pixels set to 0.")
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

def calculate_cloud_coverage(cloud_mask_path, aoi_gdf_crs_mask):
    try:
        with rasterio.open(cloud_mask_path) as cloud_src:
            if not cloud_src.crs:
                logging.error(f"CRS is missing for cloud mask {cloud_mask_path}. Cannot calculate cloud coverage.")
                return 1.0

            aoi_geometry_transformed = aoi_gdf_crs_mask.geometry.iloc[0]
            try:
                mask = features.geometry_mask([aoi_geometry_transformed], out_shape=cloud_src.shape, 
                                           transform=cloud_src.transform, invert=True, all_touched=True)
                cloud_data = cloud_src.read(1, masked=False)
                cloud_data_aoi = cloud_data[mask]
                if cloud_data_aoi.size == 0:
                    logging.warning(f"AOI mask resulted in zero pixels for cloud mask {cloud_mask_path}. "
                                    f"Assuming 0% cloud in intersection.")
                    return 0.0
                cloudy_pixels = np.sum(cloud_data_aoi > 30)
                return cloudy_pixels / cloud_data_aoi.size
            except ValueError:
                logging.warning(f"AOI geometry likely outside bounds of cloud mask {cloud_mask_path}. "
                                f"Assuming 100% cloud for safety.")
                return 1.0
            except Exception as e_cloud_mask:
                logging.error(f"Error during cloud mask processing for {cloud_mask_path}: {e_cloud_mask}\n{traceback.format_exc()}")
                return 1.0
    except rasterio.RasterioIOError as e_rio: logging.error(f"Rasterio error opening cloud mask {cloud_mask_path}: {e_rio}")
    except Exception as e: logging.error(f"Error calculating cloud cover for {cloud_mask_path}: {e}")
    return 1.0

def check_image_suitability(geo_coverage, valid_pix_perc, eff_coverage, cloud_perc):
    from .configuration import COMPLEMENT_IMAGE_MIN_GEOGRAPHIC_COVERAGE_THRESHOLD, MAX_CLOUD_COVERAGE_THRESHOLD
    if valid_pix_perc <= 1e-6: return False, f"IMAGEM SEM PIXELS VÁLIDOS NA AOI ({valid_pix_perc:.2%})"
    if geo_coverage < COMPLEMENT_IMAGE_MIN_GEOGRAPHIC_COVERAGE_THRESHOLD: 
        return False, f"IMAGEM COM COBERTURA GEOGRÁFICA INSUFICIENTE ({geo_coverage:.2%} < " \
                      f"{COMPLEMENT_IMAGE_MIN_GEOGRAPHIC_COVERAGE_THRESHOLD:.0%})"
    min_effective_coverage_required = COMPLEMENT_IMAGE_MIN_GEOGRAPHIC_COVERAGE_THRESHOLD * 0.5
    if eff_coverage < min_effective_coverage_required: 
        return False, f"IMAGEM COM COBERTURA EFETIVA INSUFICIENTE ({eff_coverage:.2%} < " \
                      f"{min_effective_coverage_required:.0%})"
    if cloud_perc > MAX_CLOUD_COVERAGE_THRESHOLD: 
        return False, f"IMAGEM REJEITADA: Muitas nuvens ({cloud_perc:.1%} > {MAX_CLOUD_COVERAGE_THRESHOLD:.0%})"
    return True, "OK"

def classify_image(effective_coverage):
    from .configuration import CENTRAL_IMAGE_EFFECTIVE_COVERAGE_THRESHOLD
    return "central" if effective_coverage >= CENTRAL_IMAGE_EFFECTIVE_COVERAGE_THRESHOLD else "complement"