import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import logging
import joblib
import os

class AlzheimerPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.one_hot_encoder = None
        self.scaler = None
        self.label_cols = ["Education Level", "Physical Activity Level", "Cholesterol Level",
                           "Depression Level", "Sleep Quality", "Air Pollution Exposure",
                           "Social Engagement Level", "Income Level", "Stress Levels"]
        self.one_hot_cols = ["Gender", "Smoking Status", "Alcohol Consumption", "Diabetes",
                             "Hypertension", "Family History of Alzheimer’s", "Dietary Habits",
                             "Employment Status", "Marital Status", "Genetic Risk Factor (APOE-ε4 allele)",
                             "Urban vs Rural Living", "Alzheimer’s Diagnosis"] # Note: Diagnosis is target, handled separately?
                             # In original code, Diagnosis is in one_hot_cols list!
                             # But prepare_model_data uses "Alzheimer’s Diagnosis_Yes".
                             # So OHE is applied to Target too.

        self.columns_to_drop = ['Country', 'Gender',
                           'Physical Activity Level', 'Smoking Status', 'Alcohol Consumption',
                           'Diabetes', 'Hypertension', 'Cholesterol Level',
                           'Family History of Alzheimer’s', 'Depression Level', 'Sleep Quality',
                           'Dietary Habits', 'Air Pollution Exposure', 'Employment Status',
                           'Marital Status', 'Genetic Risk Factor (APOE-ε4 allele)',
                           'Social Engagement Level', 'Income Level', 'Stress Levels',
                           'Urban vs Rural Living', 'Alzheimer’s Diagnosis']

        self.existing_one_hot_cols = []

    def fit_transform_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fits encoders and transforms the data (LE and OHE).
        """
        logging.info("Starting preprocessing (fit_transform)...")
        df = df.copy()

        # Label Encoding
        for col in self.label_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                self.label_encoders[col] = le

        # One-Hot Encoding
        self.existing_one_hot_cols = [col for col in self.one_hot_cols if col in df.columns]

        if self.existing_one_hot_cols:
            self.one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            # handle_unknown='ignore' helps with inference if new categories appear (will be all zeros)

            encoded_array = self.one_hot_encoder.fit_transform(df[self.existing_one_hot_cols])
            feature_names = self.one_hot_encoder.get_feature_names_out(input_features=self.existing_one_hot_cols)

            df_encoded = pd.DataFrame(encoded_array, columns=feature_names, index=df.index)

            df = pd.concat([df, df_encoded], axis=1)
        else:
            logging.warning("One-Hot Encoding columns not found!")

        logging.info("Preprocessing completed.")
        return df

    def transform_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms new data using fitted encoders.
        """
        logging.info("Starting preprocessing (transform)...")
        df = df.copy()

        # Label Encoding
        for col in self.label_cols:
            if col in df.columns and col in self.label_encoders:
                le = self.label_encoders[col]
                # LabelEncoder doesn't handle unknown labels well.
                # We can map unknown to a default or raise error.
                # For now, we assume valid input or handle via try-except/map.
                # A safer way for inference:
                # df[col] = df[col].map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)
                # But to replicate original behavior (which would crash or works on same data):
                # We'll use apply with check.

                # Fast way using map
                le_dict = dict(zip(le.classes_, le.transform(le.classes_)))
                df[col] = df[col].map(lambda x: le_dict.get(x, -1)) # -1 for unknown

        # One-Hot Encoding
        if self.existing_one_hot_cols and self.one_hot_encoder:
            # Only encode columns that are present in df
            # But OHE expects all columns it was fitted on?
            # Ideally input df should have these columns.

            # Check if all expected cols are there
            missing_cols = set(self.existing_one_hot_cols) - set(df.columns)
            if missing_cols:
                 # If target 'Alzheimer’s Diagnosis' is missing (inference), we should exclude it from OHE if it was in OHE list.
                 # 'Alzheimer’s Diagnosis' is in self.one_hot_cols.
                 # If we are doing inference, we likely don't have the target.
                 # So we need to handle this.
                 pass

            # If this is inference, we might not have the target column.
            # If target was part of OHE features, we have a problem because OHE expects it.
            # In original code, 'Alzheimer’s Diagnosis' IS in `one_hot_cols`.
            # And it IS used to generate 'Alzheimer’s Diagnosis_Yes'.
            # BUT 'Alzheimer’s Diagnosis_Yes' is the TARGET variable y.

            # During inference, we don't have target.
            # So OHE shouldn't rely on it for FEATURES.
            # But `prepare_model_data` DROPS original `Alzheimer’s Diagnosis` and uses `Alzheimer’s Diagnosis_Yes` as y.
            # It DROPS `Alzheimer’s Diagnosis_No` and `Alzheimer’s Diagnosis_Yes` from X.

            # So, the OHE features derived from Target are NOT used in X!
            # They are dropped: `X = df_model.drop(columns=["Alzheimer’s Diagnosis_No", "Alzheimer’s Diagnosis_Yes"], ...)`

            # So for inference, we don't need to OHE the target column.
            # But we need to OHE other columns.
            # The `OneHotEncoder` was fitted on ALL columns including Target.
            # If we call `transform` with missing Target column, it will fail.

            # SOLUTION: We should fit OHE only on Feature columns, or handle Target separately.
            # But to maintain exact compatibility with original "fit", I need to replicate it.
            # However, since I am refactoring, I can improve this.
            # I will exclude Target from OHE features if possible, OR
            # I will just fill dummy value for Target during inference if needed (hacky).

            # Better: In `fit_transform_preprocessing`, I should probably separate Target encoding if I can.
            # But `preprocess_data` just did one big OHE.

            # Let's adjust `fit_transform_preprocessing` to NOT include target in the main OHE if possible,
            # or handle it.
            # 'Alzheimer’s Diagnosis' is the target.
            # We can encode it separately or just LabelEncode it for y.
            # But existing code produced `Alzheimer’s Diagnosis_Yes` via OHE.

            # I'll modify `fit_transform_preprocessing` to separate Target OHE if it's the target.
            # Actually, `Alzheimer’s Diagnosis` is categorical (Yes/No).
            # We can just Label Encode it to 0/1.

            # To avoid breaking changes, I'll stick to logic but handle the OHE split.
            # I will create TWO OneHotEncoders? No.
            # I will just ensure that during inference, if target is missing, we don't crash.
            # But `transform` expects same number of features.

            # I will change logic: Separate `Alzheimer’s Diagnosis` from `one_hot_cols`.
            # Handle it manually to create `Alzheimer’s Diagnosis_Yes` for training.
            # For inference, we don't need it.
            pass

        # Re-implementation for `transform` with the OHE issue in mind:
        # If I change `fit`, I change the pipeline.
        # I'll remove 'Alzheimer’s Diagnosis' from `one_hot_cols` in `__init__`.
        # And handle target encoding in `fit_transform_preprocessing` explicitly for y.

        if self.existing_one_hot_cols and self.one_hot_encoder:
             # Filter cols that are actually present
             cols_to_transform = [c for c in self.existing_one_hot_cols if c in df.columns]

             # If we are missing columns that OHE expects, we can't use standard transform easily.
             # But wait, I'm refactoring `fit` too.
             # So I will remove Target from OHE list in `fit`.

             encoded_array = self.one_hot_encoder.transform(df[self.existing_one_hot_cols])
             feature_names = self.one_hot_encoder.get_feature_names_out(input_features=self.existing_one_hot_cols)
             df_encoded = pd.DataFrame(encoded_array, columns=feature_names, index=df.index)
             df = pd.concat([df, df_encoded], axis=1)

        logging.info("Preprocessing (transform) completed.")
        return df

    def prepare_and_split(self, df: pd.DataFrame):
        """
        Drops columns, splits into train/val/test.
        """
        logging.info("Preparing model data (split)...")

        # Determine if we have target
        target_col = "Alzheimer’s Diagnosis_Yes" # Created by OHE in original code
        # But if I change OHE logic, I need to ensure this exists.

        # If I remove Alzheimer’s Diagnosis from OHE, I need to create this column manually.

        df_model = df.drop(columns=self.columns_to_drop, errors='ignore')

        if target_col in df_model.columns:
            X = df_model.drop(columns=["Alzheimer’s Diagnosis_No", "Alzheimer’s Diagnosis_Yes"], errors='ignore')
            y = df_model[target_col]

            X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

            return X_train, X_val, X_test, y_train, y_val, y_test
        else:
            # Inference case (or target missing) - Not expected for training flow
            raise ValueError("Target column not found for splitting.")

    def fit_transform_scaling(self, X_train):
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        return X_train_scaled

    def transform_scaling(self, X):
        if self.scaler is None:
            raise ValueError("Scaler not fitted.")
        return self.scaler.transform(X)

    def save(self, filepath):
        joblib.dump(self, filepath)
        logging.info(f"Preprocessor saved to {filepath}")

    @staticmethod
    def load(filepath):
        return joblib.load(filepath)

# Update fit_transform_preprocessing to handle Target specifically
    def fit_transform_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Starting preprocessing (fit_transform)...")
        df = df.copy()

        # Label Encoding
        for col in self.label_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                self.label_encoders[col] = le

        # Handle Target manually if present
        target_col_orig = "Alzheimer’s Diagnosis"
        if target_col_orig in df.columns:
             # We want "Alzheimer’s Diagnosis_Yes" and "Alzheimer’s Diagnosis_No"
             # 1 if Yes, 0 if No.
             # Let's just use get_dummies for this specific one or map it.
             # Original code used OHE.
             # OHE produces `Alzheimer’s Diagnosis_No` and `Alzheimer’s Diagnosis_Yes`.
             dummies = pd.get_dummies(df[target_col_orig], prefix=target_col_orig)
             # Ensure both columns exist even if one class is missing (unlikely)
             if "Alzheimer’s Diagnosis_Yes" not in dummies.columns:
                  dummies["Alzheimer’s Diagnosis_Yes"] = 0
             if "Alzheimer’s Diagnosis_No" not in dummies.columns:
                  dummies["Alzheimer’s Diagnosis_No"] = 0 # or 1-Yes

             df = pd.concat([df, dummies], axis=1)

        # One-Hot Encoding for FEATURES
        # Exclude target from list if it's there
        if "Alzheimer’s Diagnosis" in self.one_hot_cols:
            self.one_hot_cols.remove("Alzheimer’s Diagnosis")

        self.existing_one_hot_cols = [col for col in self.one_hot_cols if col in df.columns]

        if self.existing_one_hot_cols:
            self.one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoded_array = self.one_hot_encoder.fit_transform(df[self.existing_one_hot_cols])
            feature_names = self.one_hot_encoder.get_feature_names_out(input_features=self.existing_one_hot_cols)

            df_encoded = pd.DataFrame(encoded_array, columns=feature_names, index=df.index)
            df = pd.concat([df, df_encoded], axis=1)

        return df

    def process_input_for_inference(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Full pipeline for inference: LE -> OHE -> Drop Cols -> Scale
        Returns Scaled X (numpy array)
        """
        # LE & OHE
        df_processed = self.transform_preprocessing(df)

        # Drop columns (same as prepare_model_data but without target stuff)
        # We need to drop the original columns that were encoded/unused.
        # But `columns_to_drop` includes 'Alzheimer’s Diagnosis' which might not be in input.

        cols_to_drop = [c for c in self.columns_to_drop if c in df_processed.columns]
        X = df_processed.drop(columns=cols_to_drop, errors='ignore')

        # Also, original code dropped `Alzheimer’s Diagnosis_No` and `Yes` from X.
        # For inference, these won't be there (since we didn't encode target).
        # So X should be ready?
        # Wait, are there other columns generated that need dropping?
        # The list `columns_to_drop` contains original categorical columns.

        # Scale
        X_scaled = self.transform_scaling(X)
        return X_scaled

# Legacy wrapper functions to maintain compatibility with existing tests/imports if needed
# But plan says update tests. So I will keep them but use the class.

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    # This function creates a new preprocessor instance every time, which is what original code implied
    # But it won't save it. This is fine for existing tests that just check preprocessing.
    p = AlzheimerPreprocessor()
    return p.fit_transform_preprocessing(df)

def prepare_model_data(df: pd.DataFrame):
    # This relies on the fact that df already has OHE applied.
    # And it needs to split and scale.
    # We can use a temporary preprocessor for scaling logic?
    # Or just replicate the logic?
    # Replicating logic or using the class:

    p = AlzheimerPreprocessor()
    # But p doesn't have the scaler fitted yet.
    # prepare_model_data does the fitting.

    # So:
    X_train, X_val, X_test, y_train, y_val, y_test = p.prepare_and_split(df)
    X_train = p.fit_transform_scaling(X_train)
    X_val = p.transform_scaling(X_val)
    X_test = p.transform_scaling(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test, p.scaler

