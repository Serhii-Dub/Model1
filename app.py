import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions

# Завантажуємо попередньо натреновану модель ResNet50
model = ResNet50(weights='imagenet')

# Завантажуємо зображення
img_path = 'path_to_image.jpg'  # Вкажіть шлях до зображення
img = image.load_img(img_path, target_size=(224, 224))

# Перетворюємо зображення в тензор
img_array = image.img_to_array(img)

# Додаємо розмірність, необхідну для моделі
img_array = np.expand_dims(img_array, axis=0)

# Попередня обробка зображення для ResNet50
img_array = preprocess_input(img_array)

# Передбачення
predictions = model.predict(img_array)

# Розшифровка результатів (отримуємо назви класів)
decoded_predictions = decode_predictions(predictions, top=3)[0]

# Виводимо топ-3 передбачення
for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
    print(f"{i+1}: {label} ({score:.2f})")
