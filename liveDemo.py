import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import sys
import os

# Cargar modelo previamente entrenado
MODEL_PATH = "Toto_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Diccionario de clases (ajústalo si cambian)
class_labels = ['Dark', 'Green', 'Light', 'Medium']

def preprocess_image(img_path, target_size=(224, 224)):
    """Carga y prepara una imagen para predicción."""
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # normalización
    return np.expand_dims(img_array, axis=0)

def predict_roast(img_path):
    if not os.path.exists(img_path):
        print(f"Error: No se encontró la imagen en {img_path}")
        return
    
    img_processed = preprocess_image(img_path)
    prediction = model.predict(img_processed)
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction)
    
    print(f"\n📷 Imagen: {img_path}")
    print(f"🔎 Predicción: {predicted_class}")
    print(f"📊 Confianza: {confidence:.2f}")

# Uso por consola:
# python predict_coffee_roast.py ruta_a_imagen.jpg
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python predict_coffee_roast.py <ruta_a_imagen>")
    else:
        predict_roast(sys.argv[1])
