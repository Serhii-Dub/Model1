from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

def build_model(num_classes=10):
    # Завантаження ResNet50 як feature extractor
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Заморожування базової моделі
    for layer in base_model.layers:
        layer.trainable = False

    # Додавання кастомних шарів
    x = Flatten()(base_model.output)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)  # Dropout для запобігання перенавчанню
    x = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=x)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Генерація даних для навчання
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    preprocessing_function=preprocess_input
)

X_train = np.random.rand(100, 224, 224, 3)
y_train = np.eye(10)[np.random.randint(0, 10, 100)]

train_generator = datagen.flow(X_train, y_train, batch_size=10)

# Побудова та навчання моделі
model = build_model(num_classes=10)
model.fit(train_generator, epochs=5)

# Збереження моделі
model.save('model/model.h5')
print("Модель збережено.")
