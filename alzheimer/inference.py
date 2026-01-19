# -*- coding: utf-8 -*-
"""
Alzheimer Inference Script
--------------------------
This script loads a trained model and preprocessor to make predictions on new data.
"""

import argparse
import logging
import sys
import os
import pandas as pd
import joblib

# Add the current directory to path so we can import src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import AlzheimerPreprocessor
from src.model import load_model

def main():
    parser = argparse.ArgumentParser(description="Alzheimer Inference")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model (pkl)")
    parser.add_argument("--preprocessor_path", type=str, required=True, help="Path to saved preprocessor (pkl)")
    parser.add_argument("--output_path", type=str, default="predictions.csv", help="Path to save predictions")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

    input_path = args.input_path
    model_path = args.model_path
    preprocessor_path = args.preprocessor_path
    output_path = args.output_path

    if not os.path.exists(input_path):
        logging.error(f"Input file not found: {input_path}")
        sys.exit(1)

    if not os.path.exists(model_path):
        logging.error(f"Model file not found: {model_path}")
        sys.exit(1)

    if not os.path.exists(preprocessor_path):
        logging.error(f"Preprocessor file not found: {preprocessor_path}")
        sys.exit(1)

    try:
        logging.info(f"Loading data from {input_path}...")
        df = pd.read_csv(input_path)

        logging.info(f"Loading preprocessor from {preprocessor_path}...")
        preprocessor = AlzheimerPreprocessor.load(preprocessor_path)

        logging.info(f"Loading model from {model_path}...")
        model = load_model(model_path)

        logging.info("Preprocessing data...")
        X = preprocessor.process_input_for_inference(df)

        logging.info("Making predictions...")
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1] # Probability of Class 1 (Yes)

        df_output = df.copy()
        df_output['Predicted_Alzheimers'] = predictions
        df_output['Probability'] = probabilities

        # Convert 1/0 to Yes/No if desired
        df_output['Predicted_Alzheimers_Label'] = df_output['Predicted_Alzheimers'].map({1: 'Yes', 0: 'No'})

        logging.info(f"Saving predictions to {output_path}...")
        df_output.to_csv(output_path, index=False)

        logging.info("Inference completed successfully.")

    except Exception as e:
        logging.error(f"An error occurred during inference: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()
