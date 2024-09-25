
import streamlit as st
import joblib
from joblib import load
import pandas as pd

# Modelinizi yükleyin
try:
    model = load('xgb_model.joblib')
except Exception as e:
    st.error("Model yüklenirken bir hata oluştu: {}".format(e))

# Kullanıcıdan giriş alma
st.title("Obezite Tahmin Modeli")

# Kullanıcıdan girdi alma
sex = st.selectbox("Cinsiyet(1 : Erkek , 2 : Kadın):", [1, 2])  
age = st.number_input("Yaş:", min_value=0, max_value=120)
height = st.number_input("Boy (cm):", min_value=0, max_value=250)
overweight_obese_family = st.selectbox("Ailede Obezite Geçmişi(1 : Obeziteye sahip aile bireyi var, 2 : Obeziteye sahip aile bireyi yok):", [1, 2])  
consumption_of_fast_food = st.selectbox("Hızlı Gıda Tüketimi (1 : Evet ,2 : Hayır):", [1, 2])
frequency_of_consuming_vegetables = st.selectbox("Sebze Tüketim Sıklığı (1 : Nadiren, 2 : Ara sıra , 3 : Her zaman):", [1, 2, 3])
number_of_main_meals_daily = st.selectbox("Günlük Ana Yemek Sayısı (1 : Günde 1-2 , 2 : Günde 3 , 3: Günde 3 'ten fazla):", [1, 2, 3])
food_intake_between_meals = st.selectbox("Ana Öğünler Arası Gıda Tüketimi (1 :Nadiren yaparım, 2 : Ara sıra yaparım , 3 : Genellikle yaparım, 4 : Her zaman yaparım):", [1, 2, 3, 4])
smoking = st.selectbox("Sigara Kullanımı (1: Evet, 2: Hayır):", [1, 2])
liquid_intake_daily = st.selectbox("Günlük Sıvı Tüketimi (1 : 1 litreden az, 2 : 1 ile 2 litre arası , 3 : 2 litreden fazla):", [1, 2, 3])
calculation_of_calorie_intake = st.selectbox("Kalori Alımını Hesaplama (1 : Kalori alımını hesaplıyorum, 2 : Kalori alımını hesaplamaıyorum):", [1, 2])
physical_exercise = st.selectbox("Fiziksel Egzersiz (1 : Fiziksel olarak aktif değilim, 2 : Haftada 1 ile 2 gün yaparım, 3 : Haftada 3 ile 4 gün yaparım, 4 : Haftada 5 ile 6 gün yaparım, 5 : Haftada 6 veya daha fazla yaparım):", [1, 2, 3, 4, 5])
schedule_dedicated_to_technology = st.selectbox("Teknolojiye Ayırılan Zaman (1 : Günlük 0 ile 2 saat ayırırım, 2 : Günlük 3 ile 5 saat ayırırım, 3 : Günlük 5 saatten fazla ayırırım):", [1, 2, 3, 4, 5])
type_of_transportation_used = st.selectbox("Kullanılan Ulaşım Türü (1 : Otomobil , 2 : Motor , 3 : Bisiklet, 4 : Toplu Taşıma , 5 : Yürüme):", [1, 2, 3, 4, 5])

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
    predictions = model.predict(input_data)
    st.write("Tahmin edilen sınıf:", predictions[0])
    if predictions[0] == 1:
        st.write("Tahmin: Obez")
    else:
        st.write("Tahmin: Obez Değil")
