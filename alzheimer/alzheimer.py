# -*- coding: utf-8 -*-
"""
Alzheimer Detection Project
-------------------------
This project aims to develop a machine learning model for Alzheimer's disease detection.
It includes data analysis, preprocessing, model training (XGBoost with GridSearchCV), and evaluation.
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
from src.preprocessing import AlzheimerPreprocessor
from src.model import train_xgb_model, save_model
from src.evaluation import evaluate_model

def main():
    parser = argparse.ArgumentParser(description="Alzheimer Detection Project")
    parser.add_argument("--filepath", type=str, default="alzheimers_prediction_dataset.csv", help="Path to the dataset CSV file")
    parser.add_argument("--output_dir", type=str, default="plots", help="Directory to save plots")
    parser.add_argument("--model_dir", type=str, default="models", help="Directory to save trained model and preprocessor")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of jobs for GridSearchCV")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

    warnings.filterwarnings('ignore')

    filepath = args.filepath
    output_dir = args.output_dir
    model_dir = args.model_dir

    logging.info(f"Using dataset: {filepath}")
    logging.info(f"Saving plots to: {output_dir}")
    logging.info(f"Saving models to: {model_dir}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    try:
        # Load Data
        df = load_data(filepath)

        # EDA
        perform_eda(df, output_dir=output_dir)

        # Initialize Preprocessor
        preprocessor = AlzheimerPreprocessor()

        # Preprocessing (LE & OHE)
        # This fits the encoders on the whole dataset
        df_processed = preprocessor.fit_transform_preprocessing(df)

        # Split Data
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.prepare_and_split(df_processed)

        # Scale Data (Fit on Train, Transform all)
        X_train = preprocessor.fit_transform_scaling(X_train)
        X_val = preprocessor.transform_scaling(X_val)
        X_test = preprocessor.transform_scaling(X_test)

        # Save Preprocessor
        preprocessor.save(os.path.join(model_dir, "preprocessor.pkl"))

        # Train Model
        best_model = train_xgb_model(X_train, y_train, n_jobs=args.n_jobs)

        # Save Model
        save_model(best_model, os.path.join(model_dir, "model.pkl"))

        # Evaluate Model
        evaluate_model(best_model, X_val, y_val, X_test, y_test, output_dir=output_dir)

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()
