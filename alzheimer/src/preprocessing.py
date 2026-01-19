import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import logging

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Veri setinde önişleme işlemleri yapar:
      - Belirli sütunlarda etiketleme (Label Encoding)
      - One-Hot Encoding uygulanması
      - Yeni özelliklerin oluşturulması
    """
    logging.info("Starting preprocessing...")
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

    # Check if columns exist before encoding
    existing_one_hot_cols = [col for col in one_hot_cols if col in df.columns]

    if existing_one_hot_cols:
        ohe = OneHotEncoder(sparse_output=False)
        df_encoded = pd.DataFrame(ohe.fit_transform(df[existing_one_hot_cols]),
                                  columns=ohe.get_feature_names_out(input_features=existing_one_hot_cols))
        # Drop original columns and concatenate encoded ones
        # Original code didn't drop original columns here, but `prepare_model_data` drops them later.
        # However, concatenating without re-indexing might cause issues if indices are messed up.
        # But here df index is likely default RangeIndex.
        df = pd.concat([df, df_encoded], axis=1)
    else:
        logging.warning("One-Hot Encoding için gerekli sütunlar veri setinde bulunamadı!")

    logging.info("Preprocessing completed.")
    return df


def prepare_model_data(df: pd.DataFrame):
    """
    Model için kullanılacak veri setini hazırlar:
      - Gereksiz sütunların çıkarılması
      - Bağımsız (X) ve bağımlı (y) değişkenlerin ayrılması
      - Veri setinin eğitim, doğrulama ve test olarak bölünmesi
      - Standartlaştırma (scaling) işlemi
    """
    logging.info("Preparing model data...")
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
    # Note: Assuming Alzheimer’s Diagnosis_Yes exists. If not, we should probably check.
    if "Alzheimer’s Diagnosis_Yes" not in df_model.columns:
         raise ValueError("Target column 'Alzheimer’s Diagnosis_Yes' not found. Preprocessing might have failed or target column missing.")

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

    logging.info("Model data prepared.")
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler
