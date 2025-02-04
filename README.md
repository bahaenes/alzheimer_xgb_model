# Alzheimer Tespit Projesi

## Proje Hakkında
Bu proje, Alzheimer hastalığını tespit etmek amacıyla geliştirilmiş bir makine öğrenmesi uygulamasıdır. Proje kapsamında, veri analizi, önişleme, model eğitimi (XGBoost sınıflandırıcısı kullanılarak GridSearchCV ile) ve model değerlendirme adımları gerçekleştirilmiştir.

## İçerik
- **Veri Analizi (EDA):**
  - Veri setinin ilk ve son satırları, genel istatistiksel özet ve veri seti hakkında bilgi.
  - Sayısal değişkenler için korelasyon analizi ve ısı haritası.
  - Kategorik değişkenler için Ki-Kare testi uygulanarak p-değer matrisi görselleştirmesi.
  
- **Veri Önişleme:**
  - Label Encoding: Eğitim, fiziksel aktivite, kolesterol, depresyon, uyku kalitesi, hava kirliliği, sosyal etkileşim, gelir ve stres seviyeleri gibi değişkenlerin kodlanması.
  - One-Hot Encoding: Cinsiyet, sigara kullanımı, alkol tüketimi, diyabet, hipertansiyon, aile geçmişi, diyet, istihdam durumu, medeni durum, genetik risk faktörü ve yaşam yeri gibi kategorik değişkenlerin işlenmesi.

- **Model Eğitimi:**
  - Model için veri, eğitim, doğrulama ve test setlerine ayrılmıştır.
  - XGBoost sınıflandırıcısı, GridSearchCV ile hiperparametre optimizasyonu yapılarak eğitilmiştir.
  
- **Model Değerlendirme:**
  - Doğruluk, sınıflandırma raporu, karmaşıklık matrisi (confusion matrix) ve ROC-AUC skorları hesaplanmıştır.
  - Modelin önemli özellikleri görselleştirilmiştir.

## Kullanılan Teknolojiler ve Kütüphaneler
- **Python 3.x**
- **Pandas** ve **NumPy**: Veri işleme ve hesaplama.
- **Seaborn** ve **Matplotlib**: Görselleştirme.
- **Scikit-Learn**: Veri önişleme, eğitim ve model değerlendirme.
- **XGBoost**: Modelleme ve GridSearchCV ile hiperparametre optimizasyonu.
- **SciPy**: İstatistiksel testler (Ki-Kare testi).

## Gereksinimler
Aşağıdaki kütüphanelerin yüklü olması gerekmektedir. Gerekli paketleri yüklemek için `requirements.txt` dosyasını kullanabilirsiniz.

```bash
pip install -r requirements.txt
git clone https://github.com/kullanici_adiniz/alzheimer-tespit-projesi.git
cd alzheimer-tespit-projesi
pip install -r requirements.txt
python alzheimer_refactored.py
alzheimer-tespit-projesi/
│
├── alzheimer_refactored.py        # Ana Python dosyası
├── alzheimers_prediction_dataset.csv  # Veri seti
├── README.md                      # Bu dokümantasyon dosyası
└── requirements.txt               # Gerekli Python paketleri listesi
