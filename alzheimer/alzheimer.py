# -*- coding: utf-8 -*-
"""
Alzheimer Tespit Projesi
-------------------------
Bu proje, Alzheimer hastalığı tespiti için bir makine öğrenmesi modeli geliştirmeyi amaçlamaktadır.
Veri analizi, önişleme, model eğitimi (XGBoost ile GridSearchCV kullanılarak) ve değerlendirme adımlarını içerir.
"""

import argparse
import logging
import warnings
import sys
import os

# Add the current directory to path so we can import src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_data
from src.eda import perform_eda
from src.preprocessing import preprocess_data, prepare_model_data
from src.model import train_xgb_model
from src.evaluation import evaluate_model

def main():
    parser = argparse.ArgumentParser(description="Alzheimer Detection Project")
    parser.add_argument("--filepath", type=str, default="alzheimers_prediction_dataset.csv", help="Path to the dataset CSV file")
    parser.add_argument("--output_dir", type=str, default="plots", help="Directory to save plots")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of jobs for GridSearchCV")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

    warnings.filterwarnings('ignore')

    filepath = args.filepath
    output_dir = args.output_dir
    logging.info(f"Using dataset: {filepath}")
    logging.info(f"Saving plots to: {output_dir}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        # Veri yükleme
        df = load_data(filepath)

        # Keşifsel Veri Analizi (EDA)
        perform_eda(df, output_dir=output_dir)

        # Veri önişleme
        df_processed = preprocess_data(df)

        # Model için veri hazırlama
        X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_model_data(df_processed)

        # Model eğitimi
        best_model = train_xgb_model(X_train, y_train, n_jobs=args.n_jobs)

        # Model değerlendirmesi
        evaluate_model(best_model, X_val, y_val, X_test, y_test, output_dir=output_dir)

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
