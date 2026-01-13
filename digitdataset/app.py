################################################################################################
import os

# --- SESSİZLEŞTİRME AYARLARI ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # TensorFlow donanım uyarılarını gizler
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # oneDNN mesajlarını kapatır

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR) # ABSL uyarılarını (Keras yükleme mesajı) gizler

import logging
import warnings
import io
import base64
import re
import numpy as np
from PIL import Image, ImageOps
from flask import Flask, request, jsonify, render_template

# Flask ve Python uyarılarını sustur
logging.getLogger('werkzeug').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')

from tensorflow import keras

################################################################################################
### Flask Uygulaması ve Model Yükleme
################################################################################################

app = Flask(__name__)

MODEL_PATH = 'models/mnist_model.h5'

model = None
if os.path.exists(MODEL_PATH):
    model = keras.models.load_model(MODEL_PATH)
else:
    print(f"HATA: {MODEL_PATH} bulunamadı!")

################################################################################################
### Görüntü İşleme (Preprocessing) Mantığı
################################################################################################ 

def preprocess_image_from_base64(b64data: str) -> np.ndarray:
    # Base64 formatındaki veriyi ayıkla ve resme dönüştür
    match = re.search(r"base64,(.*)", b64data)
    if not match:
        raise ValueError("Geçersiz resim verisi")
    
    img_bytes = base64.b64decode(match.group(1))
    img = Image.open(io.BytesIO(img_bytes)).convert('L') # Siyah-beyaz yap

    # Renkleri Tersine Çevir: Web'de genelde beyaz kağıda siyah çizilir, 
    # ancak MNIST modeli siyah arka plan üzerine beyaz rakam bekler.
    img = ImageOps.invert(img)

    # Rakamı Kırp: Resimdeki rakamı bul ve etrafındaki boşlukları at.
    arr = np.array(img)
    coords = np.column_stack(np.where(arr > 0))
    if coords.size > 0:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        arr = arr[y_min:y_max+1, x_min:x_max+1]
    else:
        arr = np.zeros((28, 28), dtype=np.uint8)

    # Ortalama ve Boyutlandırma: Rakamı 20x20 yapıp 28x28'lik merkeze koy.
    # Bu adım modelin rakamı daha iyi tanımasını sağlar.
    img = Image.fromarray(arr)
    img.thumbnail((20, 20), Image.Resampling.LANCZOS)
    canvas = Image.new('L', (28, 28), color=0)
    paste_x = (28 - img.width) // 2
    paste_y = (28 - img.height) // 2
    canvas.paste(img, (paste_x, paste_y))

    # Normalizasyon: 0-255 arası değerleri 0-1 arasına çek.
    arr = np.array(canvas).astype('float32') / 255.0
    arr = np.expand_dims(arr, axis=(0, -1)) # (1, 28, 28, 1) şekline getir
    return arr

################################################################################################
### Sunucu Yolları (Routes)
################################################################################################

@app.route('/')
def index():
    """Ana sayfayı (HTML) yükler."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Web'den gelen resmi alır ve sonucu tahmin eder."""
    if model is None:
        return jsonify({'error': 'Model dosyası bulunamadı!'}), 500
        
    try:
        data = request.get_json()
        img_data = data.get('image')
        
        # Resmi modele uygun hale getir
        x = preprocess_image_from_base64(img_data)
        
        # Tahmin yap
        probs = model.predict(x, verbose=0)[0]
        pred = int(np.argmax(probs))
        
        # Sonucu JSON olarak döndür
        return jsonify({
            'prediction': pred,
            'probabilities': [round(float(p), 4) for p in probs]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    print("\n" + "="*40)
    print("MNIST RAKAM TANIMA SERVISI CALISIYOR")
    print("Adres: http://127.0.0.1:5000")
    print("="*40 + "\n")
    
    # use_reloader=False terminalde çift log oluşmasını engeller
    app.run(debug=True, use_reloader=False)

################################################################################################