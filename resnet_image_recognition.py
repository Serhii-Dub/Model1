import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

# --- Part 1: Model Creation and Training ---

# Define constants
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 10
MODEL_SAVE_PATH = 'resnet_model.h5'

# Load ResNet50 model without top layers
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
base_model.trainable = False

# Add custom layers
def create_model():
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')  # Assume 10 classes for this example
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Assume dataset is in 'data/train' and 'data/validation'
train_dir = 'data/train'
validation_dir = 'data/validation'

data_gen = ImageDataGenerator(rescale=1.0/255.0)
train_data = data_gen.flow_from_directory(train_dir, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, class_mode='categorical')
validation_data = data_gen.flow_from_directory(validation_dir, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, class_mode='categorical')

model = create_model()

# Train the model
model.fit(train_data, validation_data=validation_data, epochs=EPOCHS)
model.save(MODEL_SAVE_PATH)

# --- Part 2: Web Interface with Flask ---

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model
model = load_model(MODEL_SAVE_PATH)
class_indices = train_data.class_indices
class_names = {v: k for k, v in class_indices.items()}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Preprocess the image
    img = load_img(filepath, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)[0]
    confidence = np.max(predictions)
    predicted_class = np.argmax(predictions)
    predicted_label = class_names[predicted_class]

    if confidence < 0.5:
        return jsonify({'result': 'I am not confident', 'confidence': float(confidence)})
    
    return jsonify({'result': predicted_label, 'confidence': float(confidence), 'group': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)