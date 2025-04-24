import zipfile
import shutil
from pathlib import Path
import logging

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