from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input  # Імпортуємо функцію попередньої обробки
import numpy as np
import os
from PIL import Image
import json
import urllib.request

app = Flask(__name__)

# Завантажуємо модель
model = load_model('model/model.h5')

# Папка для зберігання завантажених зображень
UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Завантаження списку класів ImageNet
imagenet_url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
response = urllib.request.urlopen(imagenet_url)
class_idx = json.load(response)

# Створюємо список класів за допомогою індексів (використовуємо тільки назви класів)
class_names = [class_idx[str(i)][1] for i in range(1000)]  # ImageNet містить 1000 класів

@app.route('/')
def index():
    return render_template('index.html')  # Завантажуємо ваш HTML шаблон

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file found'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Завантаження і обробка зображення
    img = Image.open(filepath)
    img = img.convert('RGB')  # Переконуємось, що зображення в форматі RGB
    img = img.resize((224, 224))  # Перетворюємо розмір зображення
    img_array = np.array(img) / 255.0  # Нормалізація
    img_array = np.expand_dims(img_array, axis=0)  # Додаємо batch dimension

    # Попередня обробка для ResNet50
    img_array = preprocess_input(img_array)  # Попередня обробка для ResNet50

    # Прогнозування
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]  # Клас з найвищою ймовірністю

    # Отримуємо назву класу з ImageNet
    class_label = class_names[predicted_class]  # Назва класу з ImageNet

    return jsonify({'prediction': str(predicted_class), 'class_name': class_label})

if __name__ == '__main__':
    app.run(debug=True)
