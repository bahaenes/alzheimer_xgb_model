# Source Code Modules

This directory contains the core logic for the Alzheimer's Detection project.

- **`data_loader.py`**: Handles loading the dataset from CSV files.
- **`eda.py`**: Contains functions for performing Exploratory Data Analysis (EDA), generating histograms, correlation matrices, and other plots.
- **`preprocessing.py`**: Defines the `AlzheimerPreprocessor` class, which manages Label Encoding, One-Hot Encoding, and Scaling. It ensures consistent preprocessing between training and inference.
- **`model.py`**: Wraps the XGBoost training logic, including Hyperparameter Tuning via GridSearchCV, and provides functions to save/load models.
- **`evaluation.py`**: Utilities for evaluating the model performance (Accuracy, ROC-AUC, Confusion Matrix, etc.) and saving evaluation plots.
