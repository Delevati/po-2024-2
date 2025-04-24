import xml.etree.ElementTree as ET
import re
from datetime import datetime
from pathlib import Path
import logging
import json
import pyproj
import rasterio
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import transform as shapely_transform
from rasterio import features

from .configuration import METADATA_DIR

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

def save_classification_metadata(output_dir: Path, classification: str | None, metrics: dict, date_obj: datetime | None, 
                                orbit: int | None, zip_filename: str):
    bounds_data = None
    if metrics.get('bounds'):
        b = metrics['bounds']
        if hasattr(b, 'left') and hasattr(b, 'bottom') and hasattr(b, 'right') and hasattr(b, 'top'):
            bounds_data = {'left': b.left, 'bottom': b.bottom, 'right': b.right, 'top': b.top}
        elif isinstance(b, dict) and all(k in b for k in ['left', 'bottom', 'right', 'top']):
            bounds_data = b  # Already a dict
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

def get_cloud_cover_in_geom(img_meta: dict, geometry_wgs84: Polygon) -> float:
    """
    Calcula a porcentagem de cobertura de nuvens dentro de uma geometria específica (em WGS84)
    usando a máscara de nuvens da imagem.
    """
    # Prioriza o caminho final, usa o caminho temporário como fallback
    cloud_mask_path_str = img_meta.get('cloud_mask_path') or img_meta.get('temp_cloud_mask_path')
    if not cloud_mask_path_str:
        logging.warning(f"No cloud mask path found for {img_meta.get('filename')} in get_cloud_cover_in_geom.")
        return 1.0  # Assume 100% de nuvens se não encontrar o caminho da máscara

    # Garante que o caminho existe
    cloud_mask_path = Path(cloud_mask_path_str)
    if not cloud_mask_path.is_absolute():
        # Tenta resolver o caminho relativo ao BASE_VOLUME se a configuração estiver disponível
        try:
            from .configuration import BASE_VOLUME
            cloud_mask_path = BASE_VOLUME / cloud_mask_path_str.lstrip('/')
        except ImportError:
            logging.warning(f"BASE_VOLUME not available in configuration. Trying path as is: {cloud_mask_path}")
        except Exception as e_vol:
            logging.warning(f"Error resolving path with BASE_VOLUME: {e_vol}. Trying path as is: {cloud_mask_path}")

    if not cloud_mask_path.exists():
        logging.warning(f"Cloud mask file not found at resolved path {cloud_mask_path} for {img_meta.get('filename')}.")
        return 1.0  # Assume 100% de nuvens se não encontrar o arquivo da máscara

    try:
        with rasterio.open(cloud_mask_path) as cloud_src:
            mask_crs = cloud_src.crs
            if not mask_crs:
                logging.warning(f"CRS missing for cloud mask {cloud_mask_path}. Cannot calculate overlap cloud cover.")
                return 1.0  # Assume 100% de nuvens se CRS estiver faltando

            try:
                # Transforma a geometria de entrada de WGS84 para o CRS da máscara de nuvens
                proj_wgs84 = pyproj.Proj('epsg:4326')
                proj_mask = pyproj.Proj(mask_crs)
                transformer_to_mask = pyproj.Transformer.from_proj(proj_wgs84, proj_mask, always_xy=True).transform
                geometry_mask_crs = shapely_transform(transformer_to_mask, geometry_wgs84)
            except Exception as e_transform:
                logging.error(f"Failed to transform overlap geometry to mask CRS {mask_crs} for {img_meta.get('filename')}: {e_transform}")
                return 1.0  # Assume 100% de nuvens em erro de transformação

            try:
                # Cria uma máscara para a geometria dentro da forma/transformação do raster de nuvens
                mask = features.geometry_mask([geometry_mask_crs], out_shape=cloud_src.shape,
                                           transform=cloud_src.transform, invert=True, all_touched=True)

                # Lê dados de nuvem apenas dentro da área mascarada
                cloud_data = cloud_src.read(1, masked=False)  # Lê toda a banda primeiro
                cloud_data_geom = cloud_data[mask]  # Aplica a máscara

                if cloud_data_geom.size == 0:
                    # Isso significa que a geometria, após a transformação, não se sobrepõe a nenhum pixel válido
                    # ou a criação da máscara falhou de forma inesperada.
                    logging.debug(f"Geometry mask resulted in zero pixels for cloud mask {cloud_mask_path} in get_cloud_cover_in_geom. Assuming 0% cloud in (non-existent) overlap.")
                    return 0.0  # Sem pixels de sobreposição significa 0% de nuvens *na geometria*

                # Conta pixels com probabilidade de nuvem > 30% (ajuste o limiar se necessário)
                cloudy_pixels = np.sum(cloud_data_geom > 30)
                cloud_percentage = cloudy_pixels / cloud_data_geom.size
                return cloud_percentage

            except ValueError as ve:
                # Frequentemente acontece se a geometria estiver fora dos limites do raster após a transformação
                logging.debug(f"Geometry likely outside bounds of cloud mask {cloud_mask_path} in get_cloud_cover_in_geom ({ve}). Assuming 100% cloud.")
                return 1.0
            except Exception as e_cloud_mask:
                logging.error(f"Error during cloud mask processing within geometry for {cloud_mask_path}: {e_cloud_mask}")
                return 1.0  # Assume 100% de nuvens em erro de processamento

    except rasterio.RasterioIOError as e_rio:
        logging.error(f"Rasterio error opening cloud mask {cloud_mask_path} in get_cloud_cover_in_geom: {e_rio}")
    except Exception as e:
        logging.error(f"Unexpected error calculating cloud cover within geometry for {cloud_mask_path}: {e}")

    return 1.0