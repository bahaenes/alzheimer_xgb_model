import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.stats import chi2_contingency
import logging

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

import os

def perform_eda(df: pd.DataFrame, output_dir: str = "plots"):
    """
    Veri setinin temel keşif analizini (EDA) gerçekleştirir ve bilgileri ekrana yazdırır.
    Grafikleri output_dir dizinine kaydeder.
    """
    logging.info("Starting EDA...")
    print("İlk 5 satır:")
    print(df.head())
    print("\nSon 5 satır:")
    print(df.tail())
    print("\nVeri Bilgileri:")
    df.info()
    print("\nİstatistiksel Özellikler:")
    print(df.describe().T)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Sayısal değişkenler için korelasyon ısı haritası
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(num_cols) > 0:
        plt.figure(figsize=(10, 8))
        sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm')
        plt.title("Sayısal Değişkenler Korelasyon Isı Haritası")
        plt.savefig(os.path.join(output_dir, "numeric_correlation.png"))
        plt.close()

    # Kategorik değişkenler için Ki-Kare p-değer matrisi
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    if len(cat_cols) > 0:
        chi_matrix = categorical_correlation_matrix(df, cat_cols)
        plt.figure(figsize=(15, 6))
        sns.heatmap(chi_matrix.astype(float), annot=True, cmap='coolwarm', fmt=".3f", linewidths=0.5, cbar=True)
        plt.title("Ki-Kare p-Değer Korelasyon Matrisi")
        plt.savefig(os.path.join(output_dir, "categorical_correlation.png"))
        plt.close()

    print("\nEksik Değer Sayıları:")
    print(df.isnull().sum())
    print("\nHer Sütundaki Benzersiz Değer Sayısı:")
    print(df.nunique())
    logging.info("EDA completed.")
