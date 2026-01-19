import unittest
import pandas as pd
import numpy as np
import sys
import os
from io import StringIO
import tempfile
import joblib

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import load_data
from src.preprocessing import preprocess_data, prepare_model_data, AlzheimerPreprocessor

class TestAlzheimerPipeline(unittest.TestCase):

    def setUp(self):
        # Create a dummy dataframe
        self.csv_data = """Country,Age,Gender,Education Level,BMI,Physical Activity Level,Smoking Status,Alcohol Consumption,Diabetes,Hypertension,Cholesterol Level,Family History of Alzheimer’s,Cognitive Test Score,Depression Level,Sleep Quality,Dietary Habits,Air Pollution Exposure,Employment Status,Marital Status,Genetic Risk Factor (APOE-ε4 allele),Social Engagement Level,Income Level,Stress Levels,Urban vs Rural Living,Alzheimer’s Diagnosis
Spain,90,Male,1,33.0,Medium,Never,Occasionally,No,No,Normal,No,90,Low,Poor,Healthy,High,Retired,Single,No,Low,Medium,High,Urban,No
Argentina,72,Male,7,29.9,Medium,Former,Never,No,No,Normal,No,65,Low,Good,Healthy,Medium,Unemployed,Widowed,No,High,Low,High,Urban,No
South Africa,86,Female,19,22.9,High,Current,Occasionally,No,Yes,Normal,No,43,High,Good,Average,Medium,Employed,Single,No,Low,Medium,High,Rural,Yes
China,53,Male,17,31.2,Low,Never,Regularly,Yes,No,Normal,No,81,Medium,Average,Healthy,Medium,Retired,Single,No,High,Medium,Low,Rural,No
USA,65,Female,12,25.0,High,Former,Socially,No,No,High,Yes,70,Medium,Average,Unhealthy,High,Employed,Married,Yes,Medium,High,Medium,Urban,Yes
UK,80,Male,15,28.0,Low,Never,Never,Yes,Yes,High,No,50,High,Poor,Average,Low,Retired,Widowed,No,Low,Medium,High,Rural,No
France,75,Female,10,24.0,Medium,Current,Socially,No,No,Normal,Yes,60,Low,Good,Healthy,Low,Retired,Married,No,High,Medium,Low,Urban,Yes
Spain,90,Male,1,33.0,Medium,Never,Occasionally,No,No,Normal,No,90,Low,Poor,Healthy,High,Retired,Single,No,Low,Medium,High,Urban,No
Argentina,72,Male,7,29.9,Medium,Former,Never,No,No,Normal,No,65,Low,Good,Healthy,Medium,Unemployed,Widowed,No,High,Low,High,Urban,No
South Africa,86,Female,19,22.9,High,Current,Occasionally,No,Yes,Normal,No,43,High,Good,Average,Medium,Employed,Single,No,Low,Medium,High,Rural,Yes
China,53,Male,17,31.2,Low,Never,Regularly,Yes,No,Normal,No,81,Medium,Average,Healthy,Medium,Retired,Single,No,High,Medium,Low,Rural,No
USA,65,Female,12,25.0,High,Former,Socially,No,No,High,Yes,70,Medium,Average,Unhealthy,High,Employed,Married,Yes,Medium,High,Medium,Urban,Yes
UK,80,Male,15,28.0,Low,Never,Never,Yes,Yes,High,No,50,High,Poor,Average,Low,Retired,Widowed,No,Low,Medium,High,Rural,No
France,75,Female,10,24.0,Medium,Current,Socially,No,No,Normal,Yes,60,Low,Good,Healthy,Low,Retired,Married,No,High,Medium,Low,Urban,Yes
"""
        self.filepath = "test_data.csv"
        with open(self.filepath, "w") as f:
            f.write(self.csv_data)

    def tearDown(self):
        if os.path.exists(self.filepath):
            os.remove(self.filepath)

    def test_load_data(self):
        df = load_data(self.filepath)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 14)

    def test_preprocess_data(self):
        df = load_data(self.filepath)
        df_processed = preprocess_data(df)

        # Check if new columns from one-hot encoding exist
        self.assertIn("Gender_Male", df_processed.columns)
        self.assertIn("Gender_Female", df_processed.columns)

        # Check label encoding
        self.assertTrue(pd.api.types.is_integer_dtype(df_processed["Physical Activity Level"]) or pd.api.types.is_float_dtype(df_processed["Physical Activity Level"]))

    def test_prepare_model_data(self):
        df = load_data(self.filepath)
        df_processed = preprocess_data(df)
        X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_model_data(df_processed)

        self.assertEqual(X_train.shape[1], X_val.shape[1])
        self.assertEqual(X_train.shape[1], X_test.shape[1])

        self.assertTrue(len(X_train) > 0)

    def test_preprocessor_class_flow(self):
        df = load_data(self.filepath)
        preprocessor = AlzheimerPreprocessor()

        # Fit Transform Preprocessing
        df_processed = preprocessor.fit_transform_preprocessing(df)
        self.assertIn("Gender_Male", df_processed.columns)

        # Split
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.prepare_and_split(df_processed)

        # Fit Scale
        X_train_scaled = preprocessor.fit_transform_scaling(X_train)
        self.assertEqual(X_train.shape, X_train_scaled.shape)

        # Save and Load
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            preprocessor.save(tmp.name)
            tmp_path = tmp.name

        loaded_preprocessor = AlzheimerPreprocessor.load(tmp_path)
        os.remove(tmp_path)

        # Test Inference Transform
        # Simulate new data (without target usually, but let's pass a row with target and see if it ignores/handles it)
        # Or remove target col
        df_inference = df.drop(columns=["Alzheimer’s Diagnosis"]).iloc[:1]

        # To test inference, we need to pass a dataframe.
        # But we must be careful. Existing `transform_preprocessing` implementation might crash if we don't have all columns expected?
        # My implementation of `transform_preprocessing` tries to encode columns if they exist.

        df_inf_processed = loaded_preprocessor.transform_preprocessing(df_inference)

        # Prepare for inference (drop columns etc)
        # process_input_for_inference does all steps
        X_inference = loaded_preprocessor.process_input_for_inference(df_inference)

        # Check shape matches training shape (number of features)
        self.assertEqual(X_inference.shape[1], X_train_scaled.shape[1])

if __name__ == '__main__':
    unittest.main()
