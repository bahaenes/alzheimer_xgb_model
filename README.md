# End-to-End Alzheimer's Disease Detection Project

This project implements a complete machine learning pipeline for detecting Alzheimer's disease based on various health and lifestyle factors. It covers data loading, exploratory data analysis (EDA), data preprocessing, model training (XGBoost), evaluation, and inference on new data.

## Project Overview

The goal of this project is to predict the likelihood of Alzheimer's disease diagnosis using a dataset containing patient information such as age, gender, lifestyle habits, medical history, and cognitive test scores.

The project features:
- **Exploratory Data Analysis (EDA):** Statistical summaries, correlation analysis, and visualization of categorical and numerical features.
- **Data Preprocessing:** Robust handling of categorical variables (Label Encoding, One-Hot Encoding) and scaling of numerical features using a persistent preprocessor.
- **Model Training:** Training an XGBoost Classifier with Hyperparameter Tuning (GridSearchCV).
- **Model Evaluation:** Comprehensive evaluation using Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and ROC-AUC.
- **Inference Pipeline:** A dedicated script to load the trained model and preprocessor to make predictions on new, unseen data.

## Project Structure

```
alzheimer/
├── alzheimer.py            # Main script for training and evaluation
├── inference.py            # Script for running inference on new data
├── src/                    # Source code modules
│   ├── data_loader.py      # Data loading utilities
│   ├── eda.py              # Exploratory Data Analysis functions
│   ├── preprocessing.py    # Preprocessing pipeline (AlzheimerPreprocessor class)
│   ├── model.py            # Model training and saving functions
│   └── evaluation.py       # Model evaluation and plotting
├── tests/                  # Unit tests
│   └── test_pipeline.py    # Tests for the pipeline
├── requirements.txt        # Project dependencies
└── alzheimers_prediction_dataset.csv  # Dataset file
README.md                   # Project documentation
models/                     # Directory where trained models are saved (created after training)
plots/                      # Directory where EDA and Evaluation plots are saved
```

## Installation

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

2. Install the required dependencies:
   ```bash
   pip install -r alzheimer/requirements.txt
   ```

## Usage

### 1. Training the Model

To run the full pipeline (EDA, Preprocessing, Training, Evaluation), use the `alzheimer.py` script.

```bash
python alzheimer/alzheimer.py --filepath alzheimer/alzheimers_prediction_dataset.csv --output_dir plots --model_dir models
```

**Arguments:**
- `--filepath`: Path to the dataset CSV file (default: `alzheimers_prediction_dataset.csv`).
- `--output_dir`: Directory to save plots (default: `plots`).
- `--model_dir`: Directory to save the trained model and preprocessor (default: `models`).
- `--n_jobs`: Number of parallel jobs for GridSearchCV (default: 1).
- `--verbose`: Enable verbose logging.

This command will:
1. Load the data.
2. Generate EDA plots in the `plots/` directory.
3. Preprocess the data and save the preprocessor to `models/preprocessor.pkl`.
4. Train the XGBoost model and save it to `models/model.pkl`.
5. Evaluate the model and save evaluation plots to `plots/`.

### 2. Inference (Prediction)

To make predictions on a new dataset using the trained model, use the `inference.py` script.

```bash
python alzheimer/inference.py --input_path new_data.csv --model_path models/model.pkl --preprocessor_path models/preprocessor.pkl --output_path predictions.csv
```

**Arguments:**
- `--input_path`: Path to the new input CSV file.
- `--model_path`: Path to the trained model file (`.pkl`).
- `--preprocessor_path`: Path to the saved preprocessor file (`.pkl`).
- `--output_path`: Path to save the prediction results CSV (default: `predictions.csv`).
- `--verbose`: Enable verbose logging.

The output CSV will contain the original data with appended columns:
- `Predicted_Alzheimers`: 0 or 1.
- `Probability`: Probability of the positive class (Alzheimer's Diagnosis: Yes).
- `Predicted_Alzheimers_Label`: 'Yes' or 'No'.

## Key Technologies

- **Python 3.x**
- **Pandas & NumPy:** Data manipulation.
- **Scikit-Learn:** Preprocessing, Pipeline, Evaluation.
- **XGBoost:** Gradient Boosting Machine for classification.
- **Matplotlib & Seaborn:** Data Visualization.
- **Joblib:** Model persistence.

## Testing

To run the unit tests:

```bash
python alzheimer/tests/test_pipeline.py
```

## License

This project is licensed under the MIT License.
