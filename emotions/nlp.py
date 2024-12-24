import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub

path = kagglehub.dataset_download("praveengovi/emotions-dataset-for-nlp")

print("Path to dataset files:", path)

train_df = pd.read_csv("C:/Users/BAHA ENES/.cache/kagglehub/datasets/praveengovi/emotions-dataset-for-nlp/versions/1/train.txt",header=None,names=["Text"])
test_df = pd.read_csv("C:/Users/BAHA ENES/.cache/kagglehub/datasets/praveengovi/emotions-dataset-for-nlp/versions/1/test.txt",header=None,names=["Text"])

from transformers import pipeline
import torch

# .txt dosyasını uygun formata dönüştürme
def read_txt_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            # Her satırı bir listeye ekle
            texts = [line.strip() for line in file.readlines() if line.strip()]
        return texts
    except FileNotFoundError:
        print("Dosya bulunamadı! Lütfen doğru dosya yolunu kontrol edin.")
        return []


file_path = "C:/Users/BAHA ENES/.cache/kagglehub/datasets/praveengovi/emotions-dataset-for-nlp/versions/1/train.txt"  


texts = read_txt_file(file_path)


if texts:
    # GPU'yu kontrol et ve cihazı ayarla
    device = 0 if torch.cuda.is_available() else -1  #

    
    classifier = pipeline("sentiment-analysis", device=device)

    
    results = classifier(texts)
    for text, result in zip(texts, results):
        print(f"Metin: {text}")
        print(f"Sonuç: {result['label']} (Skor: {result['score']:.2f})\n")
else:
    print("Analiz yapılacak metin bulunamadı.")

print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")