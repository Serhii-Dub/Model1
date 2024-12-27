import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import save_model
import numpy as np

# Завантаження моделі ResNet50
model = ResNet50(weights='imagenet')

# Збереження моделі у форматі .h5
save_model(model, "model99.h5")

print("Model saved as model99.h5")
