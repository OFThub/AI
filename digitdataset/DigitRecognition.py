##################################################################################################
import os

# --- SESSİZLEŞTİRME AYARLARI ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # TensorFlow donanım uyarılarını gizler
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # oneDNN mesajlarını kapatır

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR) # ABSL uyarılarını (Keras yükleme mesajı) gizler

import logging
import warnings
import cv2
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn

# Flask ve Python uyarılarını sustur
logging.getLogger('werkzeug').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')

from keras import layers, models

##################################################################################################
### 1. Veri Kümesini Yükleme (MNIST)
##################################################################################################

# Keras kütüphanesinden hazır el yazısı rakam veri setini (60.000 eğitim, 10.000 test) indiriyoruz
(X_train, y_train) , (X_test, y_test) = keras.datasets.mnist.load_data()

##################################################################################################
### 2. Veri Boyutlarını İnceleme ve Normalizasyon
##################################################################################################

print("X_train.shape : ", X_train.shape) # Eğitim görsellerinin boyutu
print("X_test.shape : ", X_test.shape)   # Test görsellerinin boyutu

# Normalizasyon: 0-255 arasındaki piksel değerlerini 0-1 arasına çekiyoruz. 
# Bu, modelin daha hızlı ve kararlı öğrenmesini sağlar.
X_train = X_train / 255.0
X_test = X_test / 255.0

##################################################################################################
### 3. Veriyi Düzleştirme (Flattening) - Sadece V1 ve V2 Modelleri İçin Gerekli
##################################################################################################

# 28x28 boyutundaki kare resimleri 784 piksellik tek bir satıra dönüştürüyoruz
X_train_flat = X_train.reshape(len(X_train), 28*28)
X_test_flat = X_test.reshape(len(X_test), 28*28)

##################################################################################################
### 4. Model Yapılandırmaları (V1, V2, V3)
##################################################################################################

# --- Model V1 (Basit Yapı): Sadece girdi ve çıktı katmanı ---
"""
model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(784,), activation='sigmoid')
])
... (Eğitim kodları)
"""

# --- Model V2 (Derin Yapı): Bir gizli katman (100 nöron) eklendi ---
"""
model = keras.Sequential([
    keras.layers.Dense(100, input_shape=(784,), activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])
... (Eğitim kodları)
"""

# --- Model V3 (CNN - En Gelişmiş): Görüntü tanıma için en iyi mimari ---
model = models.Sequential([
        # Girdi katmanı: 28x28 boyutunda 1 kanal (Siyah-Beyaz) resim
        layers.Input(shape=(28, 28, 1)),
        
        # Conv2D: Resimdeki kenarları, köşeleri ve desenleri tespit eden filtreler
        layers.Conv2D(32, 3, activation='relu'),
        # MaxPooling: Önemli bilgileri tutup resmin boyutunu küçülterek hızı artırır
        layers.MaxPooling2D(),
        
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        
        # Flatten: 2B matrisi tek boyutlu diziye çevirir (Sınıflandırma katmanına hazırlık)
        layers.Flatten(),
        
        # Tam Bağlantılı (Dense) Katmanlar
        layers.Dense(128, activation='relu'),
        # Dropout: Ezberlemeyi (overfitting) önlemek için nöronların %30'unu rastgele kapatır
        layers.Dropout(0.3),
        # Çıktı Katmanı: 10 sınıf (0-9 arası rakamlar) için olasılık dağılımı (Softmax)
        layers.Dense(10, activation='softmax')
    ])

##################################################################################################
### 5. Modeli Derleme ve Eğitme
##################################################################################################

model.compile(
        optimizer='adam', # Ağırlıkları güncelleyen optimizasyon algoritması
        loss='sparse_categorical_crossentropy', # Hata (kayıp) fonksiyonu
        metrics=['accuracy'] # Başarı kriteri: Doğruluk
    )

# Eğitimi başlat
model.fit(
        X_train, y_train,
        validation_split=0.1, # Verinin %10'unu eğitim sırasında kontrol için ayırır
        epochs=6,             # Veri setinin üzerinden 6 tam tur geç
        batch_size=128,       # Her adımda 128 resmi işle
        verbose=2             # Eğitim sürecini özet halinde göster
    )

# Test seti ile modelin genel performansını ölç
model.evaluate(X_test, y_test)

##################################################################################################
### 6. Tahmin ve Başarı Analizi (Confusion Matrix)
##################################################################################################

y_predicted = model.predict(X_test)
# En yüksek olasılıklı sınıfın indeksini al (Örn: %90 ihtimalle 5 ise sonuç 5 olur)
y_predicted_labels = [np.argmax(i) for i in y_predicted]

# Karmaşıklık Matrisi: Modelin hangi rakamı hangi rakamla karıştırdığını gösterir
cm = tf.math.confusion_matrix(labels=y_test, predictions=y_predicted_labels)

# Modeli Flask uygulamasında kullanmak üzere kaydet
model.save('models/mnist_model.h5')

# Matrisin görselleştirilmesi (Isı haritası)
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek Değer')

##################################################################################################
### 7. Harici Görüntü Tahmini (Kendi Yazdığınız Rakamlar)
##################################################################################################

image_path = 'digitdataset/1.png' 

try:
    # Görüntüyü siyah-beyaz (grayscale) olarak oku
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise FileNotFoundError(f"'{image_path}' bulunamadı.")

except FileNotFoundError as e:
    print(f"\nHATA: Dosya bulunamadı, test setinden bir örnek kullanılıyor.")
    processed_image = X_test[0] 
    
except Exception as e:
    processed_image = X_test[0]
    
else:
    # Eğer resim 28x28 değilse, modele uygun boyuta yeniden boyutlandır
    if img.shape != (28, 28):
        img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

    # Resmi normalize et
    processed_image = img.astype('float32') / 255.0
    
# Modelin beklediği formata (Batch Boyutu, Genişlik, Yükseklik) getir
image_for_prediction = processed_image.reshape(1, 28, 28) 

# Tahmin yap
predictions = model.predict(image_for_prediction)

print("\n--- Harici Görüntü Tahmini ---")
predicted_digit = np.argmax(predictions[0])
print(f"Modelin Tahmini: Rakam {predicted_digit}")

# Sonucu görselleştir
plt.figure(figsize=(4, 4))
plt.imshow(processed_image, cmap='gray')
plt.title(f"Tahmin: {predicted_digit}")

plt.show()

##################################################################################################