from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import logging
import os

def evaluate_model(model, X_val, y_val, X_test, y_test, output_dir: str = "plots"):
    """
    Eğitilmiş modelin doğrulama ve test setleri üzerindeki performansını değerlendirir ve görselleştirir.
    Grafikleri output_dir dizinine kaydeder.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logging.info("Starting model evaluation...")
    # Doğrulama seti değerlendirmesi
    y_val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    logging.info(f"Validation Accuracy: {val_acc}")
    print("Validation Accuracy:", val_acc)
    print("Validation Classification Report:\n", classification_report(y_val, y_val_pred))

    # Test seti değerlendirmesi
    y_test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    logging.info(f"Test Accuracy: {test_acc}")
    print("Test Accuracy:", test_acc)
    print("Test Classification Report:\n", classification_report(y_test, y_test_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Tahmin")
    plt.ylabel("Gerçek")
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

    # ROC-AUC hesaplanması
    try:
        y_val_probs = model.predict_proba(X_val)[:, 1]
        roc_auc = roc_auc_score(y_val, y_val_probs)
        logging.info(f"Validation ROC-AUC: {roc_auc:.4f}")
        print(f"Validation ROC-AUC: {roc_auc:.4f}")
    except AttributeError:
        logging.warning("Model does not support predict_proba, skipping ROC-AUC.")

    # Önemli özelliklerin görselleştirilmesi
    xgb.plot_importance(model)
    plt.savefig(os.path.join(output_dir, "feature_importance.png"))
    plt.close()

    logging.info("Evaluation completed.")
