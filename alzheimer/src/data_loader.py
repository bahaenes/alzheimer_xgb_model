import pandas as pd
import logging
import os

def load_data(filepath: str) -> pd.DataFrame:
    """
    Veriyi belirtilen dosya yolundan yükler.
    """
    if not os.path.exists(filepath):
        logging.error(f"File not found: {filepath}")
        raise FileNotFoundError(f"File not found: {filepath}")

    try:
        df = pd.read_csv(filepath)
        logging.info("Veri başarıyla yüklendi.")
        return df
    except Exception as e:
        logging.error(f"Veri yüklenirken hata oluştu: {e}")
        raise
