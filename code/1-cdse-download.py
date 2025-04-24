import os
import certifi
import geopandas as gpd
from cdsetool.credentials import Credentials
from cdsetool.query import query_features
from cdsetool.download import download_features
from cdsetool.monitor import StatusMonitor
from datetime import date
from dotenv import load_dotenv

load_dotenv('/Users/luryand/Documents/encode-image/coverage_otimization/code/.env')

geometry_path = "/Users/luryand/Documents/encode-image/coverage_otimization/code/APA-input/bounding-search.shp"

def file_to_wkt(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.shp':
        gdf = gpd.read_file(file_path)
    
    elif ext == '.kml':
        gdf = gpd.read_file(file_path, driver='KML')
    
    else:
        raise ValueError(f"Formato de arquivo '{ext}' não suportado. Use KML ou SHP.")
    
    return gdf.geometry.iloc[0].wkt

os.environ['SSL_CERT_FILE'] = certifi.where()
credentials = Credentials(os.getenv('CDSE_USER'), os.getenv('CDSE_PASSWORD'))

wkt_geometry = file_to_wkt(geometry_path)

features = query_features(
    "Sentinel2", 
    {
        "startDate": "2024-04-13",
        "completionDate": date(2025, 4, 13),
        "processingLevel": "S2MSI2A",
        "geometry": wkt_geometry,
    },
)

# Exibir o número de resultados encontrados
print(f"Consulta retornou {len(features)} imagens.")

# Confirmar antes de iniciar o download
if len(features) == 0:
    print("Nenhuma imagem encontrada para os critérios especificados.")
else:
    response = input("Deseja iniciar o download? (s/n): ")
    if response.lower() == 's':
        downloads = download_features(
            features,
            "/Volumes/luryand/",
            {
                "concurrency": 4,
                "monitor": StatusMonitor(),
                "credentials": credentials,
            }
        )

        for id in downloads:
            print(f"Feature {id} downloaded")
    else:
        print("Download cancelado.")