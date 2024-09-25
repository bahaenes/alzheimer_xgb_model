import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# Veriyi yükleme
df = pd.read_excel("C:/Users/BAHA ENES/archive/Obesity_Dataset.xlsx")
df.head()

# Sınıf değerlerini ayarlama
df["Class"] = df["Class"].apply(lambda x: x - 1)
X = df.drop(columns=["Class"])
y = df["Class"]

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Özellikleri standartlaştırma
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Modelin oluşturulması ve GridSearch ile en iyi hiperparametrelerin bulunması
xgb = XGBClassifier()
xgb_param = {
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7]
}
xgb_grid = GridSearchCV(xgb, xgb_param, cv=10, n_jobs=-1, verbose=2).fit(X_train, y_train)

# En iyi modelin seçilmesi ve tahmin yapılması
xgb_model = xgb_grid.best_estimator_
y_pred = xgb_model.predict(X_test)

# Model performansını değerlendirme
xgb_acc = accuracy_score(y_test, y_pred)
xgb_conf = confusion_matrix(y_test, y_pred)

print("Model Doğruluğu:", xgb_acc)
print("Karmaşıklık Matrisi:\n", xgb_conf)

# Modeli kaydetme
xgb_model.save_model('xgb_model.json')

# Streamlit arayüzü oluşturma
st.title("Obezite Tahmin Modeli")

# Kullanıcıdan giriş alma
sex = st.selectbox("Cinsiyet (1 : Erkek, 2 : Kadın):", [1, 2])  # 1: Erkek, 2: Kadın
age = st.number_input("Yaş:", min_value=0, max_value=120)
height = st.number_input("Boy (cm):", min_value=0, max_value=250)
overweight_obese_family = st.selectbox(
    "Ailede Obezite Geçmişi (1 : Obeziteye sahip aile bireyi var, 2 : Obeziteye sahip aile bireyi yok):", [1, 2])
consumption_of_fast_food = st.selectbox("Hızlı Gıda Tüketimi (1 : Evet, 2 : Hayır):", [1, 2])
frequency_of_consuming_vegetables = st.selectbox("Sebze Tüketim Sıklığı (1 : Nadiren, 2 : Ara sıra, 3 : Her zaman):",
                                                 [1, 2, 3])
number_of_main_meals_daily = st.selectbox("Günlük Ana Yemek Sayısı (1 : Günde 1-2, 2 : Günde 3, 3: Günde 3'ten fazla):",
                                          [1, 2, 3])
food_intake_between_meals = st.selectbox(
    "Ana Öğünler Arası Gıda Tüketimi (1 : Nadiren yaparım, 2 : Ara sıra yaparım, 3 : Genellikle yaparım, 4 : Her zaman yaparım):",
    [1, 2, 3, 4])
smoking = st.selectbox("Sigara Kullanımı (1: Evet, 2: Hayır):", [1, 2])
liquid_intake_daily = st.selectbox(
    "Günlük Sıvı Tüketimi (1 : 1 litreden az, 2 : 1 ile 2 litre arası, 3 : 2 litreden fazla):", [1, 2, 3])
calculation_of_calorie_intake = st.selectbox(
    "Kalori Alımını Hesaplama (1 : Kalori alımını hesaplıyorum, 2 : Kalori alımını hesaplamıyorum):", [1, 2])
physical_exercise = st.selectbox(
    "Fiziksel Egzersiz (1 : Fiziksel olarak aktif değilim, 2 : Haftada 1 ile 2 gün yaparım, 3 : Haftada 3 ile 4 gün yaparım, 4 : Haftada 5 ile 6 gün yaparım, 5 : Haftada 6 veya daha fazla yaparım):",
    [1, 2, 3, 4, 5])
schedule_dedicated_to_technology = st.selectbox(
    "Teknolojiye Ayırılan Zaman (1 : Günlük 0 ile 2 saat ayırırım, 2 : Günlük 3 ile 5 saat ayırırım, 3 : Günlük 5 saatten fazla ayırırım):",
    [1, 2, 3])
type_of_transportation_used = st.selectbox(
    "Kullanılan Ulaşım Türü (1 : Otomobil, 2 : Motor, 3 : Bisiklet, 4 : Toplu Taşıma, 5 : Yürüme):", [1, 2, 3, 4, 5])

# Giriş verilerini bir DataFrame'e dönüştürme
input_data = pd.DataFrame({
    "Sex": [sex],
    "Age": [age],
    "Height": [height],
    "Overweight_Obese_Family": [overweight_obese_family],
    "Consumption_of_Fast_Food": [consumption_of_fast_food],
    "Frequency_of_Consuming_Vegetables": [frequency_of_consuming_vegetables],
    "Number_of_Main_Meals_Daily": [number_of_main_meals_daily],
    "Food_Intake_Between_Meals": [food_intake_between_meals],
    "Smoking": [smoking],
    "Liquid_Intake_Daily": [liquid_intake_daily],
    "Calculation_of_Calorie_Intake": [calculation_of_calorie_intake],
    "Physical_Excercise": [physical_exercise],
    "Schedule_Dedicated_to_Technology": [schedule_dedicated_to_technology],
    "Type_of_Transportation_Used": [type_of_transportation_used]
})

# Modeli kullanarak tahmin yapma
if st.button("Tahmin Et"):
    # Giriş verilerini standartlaştırma
    input_data_scaled = sc.transform(input_data)

    # Tahmin yapma
    predictions = xgb_model.predict(input_data_scaled)

    # Sonuçları gösterme
    st.write("Tahmin edilen sınıf:", predictions[0])
    if predictions[0] == 1:
        st.write("Tahmin: Obez")
    else:
        st.write("Tahmin: Obez Değil")
