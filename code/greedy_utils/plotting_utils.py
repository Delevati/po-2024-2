import logging
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
import traceback
import numpy as np
import rasterio
from rasterio.warp import transform_bounds, transform
from rasterio import features
from rasterio.coords import BoundingBox
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Importar configurações e outras utils necessárias
from .configuration import PLOTS_DIR, TRASH_DIR

# --- Funções de Plotagem (Movidas do script original) ---

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

            # Preparar detalhes para plotagem
            component_details = []
            for img in [img1, img2]:
                tci_path_str = img.get('tci_path') or img.get('temp_tci_path')
                if not tci_path_str: continue # Precisa do TCI para plotar
                tci_path = Path(tci_path_str)
                if not tci_path.exists(): continue # Arquivo TCI precisa existir

                details = {
                    'filename': img.get('filename'),
                    'tci_path': str(tci_path),
                    'bounds': img.get('bounds'),
                    'crs': img.get('crs'),
                    'status': img.get('status'),
                    'reason': img.get('reason'),
                    'cloud_cov': img.get('cloud_coverage', -1),
                    'eff_cov': img.get('effective_coverage', -1),
                    'geo_cov': img.get('geographic_coverage', -1)
                }
                component_details.append(details)

            if len(component_details) == 2: # Garante que ambos os TCIs são válidos
                # Calcular cobertura efetiva combinada estimada (simples)
                eff_cov1 = component_details[0].get('eff_cov', 0)
                eff_cov2 = component_details[1].get('eff_cov', 0)
                est_combined_eff = min(1.0, eff_cov1 + eff_cov2 * 0.5) # Heurística simples

                all_bad_combinations.append({
                    'criteria_used': combo['criteria'],
                    'period_start': start_date.strftime('%Y-%m-%d'),
                    'period_end': end_date.strftime('%Y-%m-%d'),
                    'component_details': component_details,
                    'estimated_combined_effective': est_combined_eff
                })

    logging.info(f"Found {len(all_bad_combinations)} potential 'bad' mosaic examples across all periods.")
    return all_bad_combinations

def _scale_uint16_data(data: np.ma.MaskedArray, filename: str) -> np.ndarray | None:
    logging.info(f"    Scaling uint16 data for {filename} using percentile stretch.")
    try:
        valid_data = data[~data.mask]
        if valid_data.size == 0:
            logging.warning("    No valid data to calculate percentiles, converting directly.")
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
    legend_elements = [plt.Line2D([0], [0], color='black', lw=1.5, label='Área de Interesse')]
    component_labels_added = set()

    if plot_type == 'bad':
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
                         legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', label=label,
                                            markerfacecolor=color_for_legend, markersize=10, alpha=alpha))
                         component_labels_added.add(img_class)
                else: # plot_type == 'bad'
                    alpha = 0.6
                    zorder = 1

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
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    # Salvar plot
    plot_filename = output_plot_dir / f"{filename_base}.png"
    try:
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        logging.info(f"  Raster plot saved to {plot_filename}")
    except Exception as e_save:
        logging.error(f"  Failed to save raster plot {plot_filename}: {e_save}")
    finally:
        plt.close(fig)

def plot_mosaic_composition(
    mosaic_info: dict,
    metadata_lookup: dict | None,
    aoi_gdf: gpd.GeoDataFrame,
    plot_index: int,
    output_plot_dir: Path,
    plot_type: str
):
    component_details = []
    title = ""
    subtitle = ""
    filename_base = ""
    date_str = "NODATE"

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

        try:
            start_date_iso = mosaic_info.get('time_window_start')
            if start_date_iso: date_str = start_date_iso.split('T')[0]
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
            if isinstance(details['bounds'], dict) and all(k in details['bounds'] for k in ['left', 'bottom', 'right', 'top']):
                 try: details['bounds'] = BoundingBox(**details['bounds'])
                 except Exception as e_bounds: logging.warning(f"Could not convert bounds dict for {filename}: {e_bounds}"); details['bounds'] = None
            elif not isinstance(details.get('bounds'), BoundingBox):
                 logging.warning(f"Invalid bounds format for {filename}: {type(details.get('bounds'))}"); details['bounds'] = None

            component_details.append(details)

        if not component_details:
            logging.warning(f"No valid component details found for good mosaic {plot_index + 1}. Skipping plot.")
            return

        est_cov = mosaic_info.get('estimated_coverage', -1)
        avg_qual = mosaic_info.get('avg_quality_factor', -1)
        title = f"Composição de Mosaico Otimizada {plot_index + 1} ({mosaic_type_desc})"
        subtitle = f"Cob. Estimada: {est_cov:.1%} | Qualidade Média: {avg_qual:.2f} | Imagens: {len(component_filenames)}"
        filename_base = f"{date_str}_good_mosaic_{plot_index + 1}_{mosaic_type_desc}_raster"

    elif plot_type == 'bad':
        criteria_used = mosaic_info.get('criteria_used', 'unknown')
        logging.info(f"Preparing data for BAD mosaic plot {plot_index + 1} (Criteria: {criteria_used})...")
        component_details_raw = mosaic_info.get('component_details', [])
        if not component_details_raw:
            logging.warning(f"No component details found for bad mosaic {plot_index + 1}. Skipping plot.")
            return

        try:
            period_start_str = mosaic_info.get('period_start')
            if period_start_str: date_str = period_start_str
        except Exception: pass

        for detail in component_details_raw:
            if isinstance(detail.get('bounds'), dict) and all(k in detail['bounds'] for k in ['left', 'bottom', 'right', 'top']):
                 try: detail['bounds'] = BoundingBox(**detail['bounds'])
                 except Exception as e_bounds: logging.warning(f"Could not convert bounds dict for {detail.get('filename')}: {e_bounds}"); detail['bounds'] = None
            elif not isinstance(detail.get('bounds'), BoundingBox):
                 logging.warning(f"Invalid bounds format for {detail.get('filename')}: {type(detail.get('bounds'))}"); detail['bounds'] = None
            component_details.append(detail)

        criteria_map = {'highest_cloud': 'NuvensAltas', 'lowest_geo_coverage': 'CobGeoBaixa', 'rejected_pair': 'ParRejeitado', 'single_rejected_plus_worst': 'RejeitadoPiorCob'}
        criteria_desc_short = criteria_map.get(criteria_used, criteria_used)
        title = f"Exemplo de Mosaico Subótimo {plot_index + 1} (Critério: {criteria_desc_short})"
        period_start = mosaic_info.get('period_start', 'N/A')
        period_end = mosaic_info.get('period_end', 'N/A')
        est_eff_cov = mosaic_info.get('estimated_combined_effective', -1.0)
        subtitle = f"Período: {period_start} a {period_end} | Cob. Efetiva Estimada: {est_eff_cov:.1%}"
        filename_base = f"{date_str}_bad_example_{plot_index + 1}_{criteria_desc_short}_raster"

    else:
        logging.error(f"Invalid plot_type '{plot_type}' provided to plot_mosaic_composition.")
        return

    valid_component_details = [d for d in component_details if isinstance(d.get('bounds'), BoundingBox)]
    if not valid_component_details:
        logging.warning(f"No components with valid bounds found for plot {filename_base}. Skipping plot.")
        return

    _plot_mosaic_raster(
        component_details=valid_component_details,
        aoi_gdf=aoi_gdf,
        plot_index=plot_index,
        output_plot_dir=output_plot_dir,
        plot_type=plot_type,
        title=title,
        subtitle=subtitle,
        filename_base=filename_base
    )