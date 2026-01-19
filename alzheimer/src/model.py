import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import logging
import joblib

def train_xgb_model(X_train, y_train, param_grid=None, n_jobs=1):
    """
    XGBoost modelini GridSearchCV kullanarak eğitir ve en iyi modeli döner.
    """
    logging.info("Starting model training...")
    xgb_model = xgb.XGBClassifier(objective="binary:logistic",
                                  eval_metric="logloss", use_label_encoder=False)

    if param_grid is None:
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.3],
            "subsample": [0.8, 1],
            "colsample_bytree": [0.8, 1]
        }

    logging.info(f"Hyperparameter tuning with grid: {param_grid}")

    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid,
                               scoring="accuracy", cv=5, verbose=1, n_jobs=n_jobs)
    grid_search.fit(X_train, y_train)
    logging.info(f"En iyi parametreler: {grid_search.best_params_}")

    best_params = grid_search.best_params_
    best_xgb = xgb.XGBClassifier(**best_params, objective="binary:logistic",
                                 eval_metric="logloss", use_label_encoder=False)
    best_xgb.fit(X_train, y_train)
    logging.info("Model training completed.")
    return best_xgb

def save_model(model, filepath):
    joblib.dump(model, filepath)
    logging.info(f"Model saved to {filepath}")

def load_model(filepath):
    model = joblib.load(filepath)
    logging.info(f"Model loaded from {filepath}")
    return model
