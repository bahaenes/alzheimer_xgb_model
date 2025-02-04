# alzheimer_refactored.py
# -*- coding: utf-8 -*-
"""
Alzheimer Tespit Projesi
-------------------------
Bu proje, Alzheimer hastalığı tespiti için bir makine öğrenmesi modeli geliştirmeyi amaçlamaktadır.
Veri analizi, önişleme, model eğitimi (XGBoost ile GridSearchCV kullanılarak) ve değerlendirme adımlarını içerir.
"""

import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.stats import chi2_contingency
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb

warnings.filterwarnings('ignore')


def load_data(filepath: str) -> pd.DataFrame:
    """
    Veriyi belirtilen dosya yolundan yükler.
    """
    try:
        df = pd.read_csv(filepath)
        print("Veri başarıyla yüklendi.")
        return df
    except Exception as e:
        print(f"Veri yüklenirken hata oluştu: {e}")
        raise


def perform_eda(df: pd.DataFrame):
    """
    Veri setinin temel keşif analizini (EDA) gerçekleştirir ve bilgileri ekrana yazdırır.
    """
    print("İlk 5 satır:")
    print(df.head())
    print("\nSon 5 satır:")
    print(df.tail())
    print("\nVeri Bilgileri:")
    df.info()
    print("\nİstatistiksel Özellikler:")
    print(df.describe().T)

    # Sayısal değişkenler için korelasyon ısı haritası
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm')
    plt.title("Sayısal Değişkenler Korelasyon Isı Haritası")
    plt.show()

    # Kategorik değişkenler için Ki-Kare p-değer matrisi
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    chi_matrix = categorical_correlation_matrix(df, cat_cols)
    plt.figure(figsize=(15, 6))
    sns.heatmap(chi_matrix.astype(float), annot=True, cmap='coolwarm', fmt=".3f", linewidths=0.5, cbar=True)
    plt.title("Ki-Kare p-Değer Korelasyon Matrisi")
    plt.show()

    print("\nEksik Değer Sayıları:")
    print(df.isnull().sum())
    print("\nHer Sütundaki Benzersiz Değer Sayısı:")
    print(df.nunique())


def chi_square_test(x, y) -> float:
    """
    İki kategorik değişken için Ki-Kare testini uygular ve p-değerini döner.
    """
    table = pd.crosstab(x, y)
    chi2, p, dof, expected = chi2_contingency(table)
    return p


def categorical_correlation_matrix(df: pd.DataFrame, cat_cols: list) -> pd.DataFrame:
    """
    Tüm kategorik değişkenler için Ki-Kare p-değer matrisini oluşturur.
    """
    corr_matrix = pd.DataFrame(np.ones((len(cat_cols), len(cat_cols))),
                               index=cat_cols, columns=cat_cols)
    for col1, col2 in combinations(cat_cols, 2):
        p_value = chi_square_test(df[col1], df[col2])
        corr_matrix.loc[col1, col2] = p_value
        corr_matrix.loc[col2, col1] = p_value  # Simetrik matris
    return corr_matrix


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Veri setinde önişleme işlemleri yapar:
      - Belirli sütunlarda etiketleme (Label Encoding)
      - One-Hot Encoding uygulanması
      - Yeni özelliklerin oluşturulması
    """
    # Label Encoding yapılacak sütunlar
    label_cols = ["Education Level", "Physical Activity Level", "Cholesterol Level",
                  "Depression Level", "Sleep Quality", "Air Pollution Exposure",
                  "Social Engagement Level", "Income Level", "Stress Levels"]
    le = LabelEncoder()
    for col in label_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])

    # One-Hot Encoding yapılacak sütunlar
    one_hot_cols = ["Gender", "Smoking Status", "Alcohol Consumption", "Diabetes",
                    "Hypertension", "Family History of Alzheimer’s", "Dietary Habits",
                    "Employment Status", "Marital Status", "Genetic Risk Factor (APOE-ε4 allele)",
                    "Urban vs Rural Living", "Alzheimer’s Diagnosis"]
    if set(one_hot_cols).issubset(df.columns):
        ohe = OneHotEncoder(sparse_output=False)
        df_encoded = pd.DataFrame(ohe.fit_transform(df[one_hot_cols]),
                                  columns=ohe.get_feature_names_out(input_features=one_hot_cols))
        df = pd.concat([df, df_encoded], axis=1)
    else:
        print("One-Hot Encoding için gerekli sütunlar veri setinde bulunamadı!")
    return df


def prepare_model_data(df: pd.DataFrame):
    """
    Model için kullanılacak veri setini hazırlar:
      - Gereksiz sütunların çıkarılması
      - Bağımsız (X) ve bağımlı (y) değişkenlerin ayrılması
      - Veri setinin eğitim, doğrulama ve test olarak bölünmesi
      - Standartlaştırma (scaling) işlemi
    """
    # Çıkarılacak sütunlar
    columns_to_drop = ['Country', 'Gender',
                       'Physical Activity Level', 'Smoking Status', 'Alcohol Consumption',
                       'Diabetes', 'Hypertension', 'Cholesterol Level',
                       'Family History of Alzheimer’s', 'Depression Level', 'Sleep Quality',
                       'Dietary Habits', 'Air Pollution Exposure', 'Employment Status',
                       'Marital Status', 'Genetic Risk Factor (APOE-ε4 allele)',
                       'Social Engagement Level', 'Income Level', 'Stress Levels',
                       'Urban vs Rural Living', 'Alzheimer’s Diagnosis']
    df_model = df.drop(columns=columns_to_drop, errors='ignore')

    # Bağımsız değişkenler (X) ve hedef değişken (y)
    X = df_model.drop(columns=["Alzheimer’s Diagnosis_No", "Alzheimer’s Diagnosis_Yes"], errors='ignore')
    y = df_model["Alzheimer’s Diagnosis_Yes"]

    # Eğitim, doğrulama ve test bölünmesi
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3,
                                                        random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5,
                                                    random_state=42, stratify=y_temp)

    # Verinin ölçeklendirilmesi
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


def train_xgb_model(X_train, y_train):
    """
    XGBoost modelini GridSearchCV kullanarak eğitir ve en iyi modeli döner.
    """
    xgb_model = xgb.XGBClassifier(objective="binary:logistic",
                                  eval_metric="logloss", use_label_encoder=False)
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.3],
        "subsample": [0.8, 1],
        "colsample_bytree": [0.8, 1]
    }
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid,
                               scoring="accuracy", cv=5, verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print("En iyi parametreler:", grid_search.best_params_)

    best_params = grid_search.best_params_
    best_xgb = xgb.XGBClassifier(**best_params, objective="binary:logistic",
                                 eval_metric="logloss", use_label_encoder=False)
    best_xgb.fit(X_train, y_train)
    return best_xgb


def evaluate_model(model, X_val, y_val, X_test, y_test):
    """
    Eğitilmiş modelin doğrulama ve test setleri üzerindeki performansını değerlendirir ve görselleştirir.
    """
    # Doğrulama seti değerlendirmesi
    y_val_pred = model.predict(X_val)
    print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
    print("Validation Classification Report:\n", classification_report(y_val, y_val_pred))

    # Test seti değerlendirmesi
    y_test_pred = model.predict(X_test)
    print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
    print("Test Classification Report:\n", classification_report(y_test, y_test_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Tahmin")
    plt.ylabel("Gerçek")
    plt.show()

    # ROC-AUC hesaplanması
    y_val_probs = model.predict_proba(X_val)[:, 1]
    roc_auc = roc_auc_score(y_val, y_val_probs)
    print(f"Validation ROC-AUC: {roc_auc:.4f}")

    # Önemli özelliklerin görselleştirilmesi
    xgb.plot_importance(model)
    plt.show()


def main():
    # Dosya yolunu belirtin (veri seti ile aynı dizinde olması önerilir)
    filepath = "alzheimers_prediction_dataset.csv"

    # Veri yükleme
    df = load_data(filepath)

    # Keşifsel Veri Analizi (EDA)
    perform_eda(df)

    # Veri önişleme
    df_processed = preprocess_data(df)

    # Model için veri hazırlama
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_model_data(df_processed)

    # Model eğitimi
    best_model = train_xgb_model(X_train, y_train)

    # Model değerlendirmesi
    evaluate_model(best_model, X_val, y_val, X_test, y_test)


if __name__ == '__main__':
    main()
