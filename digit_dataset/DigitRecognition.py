import cv2
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
from keras import layers, models

###############################################################################
### Load MNIST dataset
###############################################################################

(X_train, y_train) , (X_test, y_test) = keras.datasets.mnist.load_data()

###############################################################################
### Dataset shapes
###############################################################################

print("X_train.shape : ", X_train.shape)
print("X_test.shape : ", X_test.shape)
print("y_train.shape : ", y_train.shape)
print("y_test.shape : ", y_test.shape)

X_train = X_train / 255.0
X_test = X_test / 255.0

###############################################################################
### Flatten the images
###############################################################################

X_train_flat = X_train.reshape(len(X_train), 28*28)
X_test_flat = X_test.reshape(len(X_test), 28*28)
print("X_train_flat.shape : ", X_train_flat.shape)
print("X_test_flat.shape : ", X_test_flat.shape)

###############################################################################
### Build the model V1
###############################################################################

"""
model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(784,), activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train_flat, y_train, epochs=5)

model.evaluate(X_test_flat, y_test)

plt.matshow(X_test[0], cmap='gray')
x_predicted = model.predict(X_test_flat)
print("x_predicted[0] : ", x_predicted[0])
print("np.argmax(x_predicted[0]) : ", np.argmax(x_predicted[0]))

x_predicted_labels = [np.argmax(i) for i in x_predicted]
print("x_predicted_labels[:5] : ", x_predicted_labels[:5])

cm = tf.math.confusion_matrix(labels=y_test,predictions=x_predicted_labels)
print("Confusion Matrix : \n", cm)


plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
"""

###############################################################################
### Build the model V2
###############################################################################

"""
model = keras.Sequential([
    keras.layers.Dense(100, input_shape=(784,), activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train_flat, y_train, epochs=5)
model.evaluate(X_test_flat,y_test)

y_predicted = model.predict(X_test_flat)
y_predicted_labels = [np.argmax(i) for i in y_predicted]
cm = tf.math.confusion_matrix(labels=y_test,predictions=y_predicted_labels)

plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
"""

###############################################################################
### Build the model V3
###############################################################################

model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(10, activation='softmax')
    ])

model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=6,
        batch_size=128,
        verbose=2
    )
model.evaluate(X_test, y_test)

y_predicted = model.predict(X_test)
y_predicted_labels = [np.argmax(i) for i in y_predicted]
cm = tf.math.confusion_matrix(labels=y_test,predictions=y_predicted_labels)

model.save('models/mnist_model.h5')

plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')

###############################################################################
### External Image Prediction
###############################################################################

image_path = 'number_dataset/1.png' 

try:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise FileNotFoundError(f"'{image_path}' bulunamadı veya yüklenemedi.")

except FileNotFoundError as e:
    print(f"\nHATA: Görüntü yüklenemedi. Dosya yolu kontrol edin: {e}")
    print("\n'1.png' bulunamadığı için Test Setindeki 0. örneği kullanıyoruz.")
    processed_image = X_test[0] 
    
except Exception as e:
    print(f"Bilinmeyen bir hata oluştu: {e}")
    processed_image = X_test[0]
    
else:
    if img.shape != (28, 28):
        img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

    processed_image = img.astype('float32') / 255.0
    
image_for_prediction = processed_image.reshape(1, 28, 28) 

predictions = model.predict(image_for_prediction)

print("\n--- Harici Görüntü Tahmini ---")
print(f"Hazırlanan görüntünün boyutu: {processed_image.shape}")
print("Modelin Olasılık Tahminleri (0-9 arası):")
print(predictions[0])

predicted_digit = np.argmax(predictions[0])

print(f"\nModelin Tahmini: Rakam {predicted_digit}")

plt.figure(figsize=(4, 4))
plt.imshow(processed_image, cmap='gray')
plt.title(f"Tahmin: {predicted_digit}")


plt.show()