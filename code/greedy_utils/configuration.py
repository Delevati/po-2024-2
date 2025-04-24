from pathlib import Path
import logging
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