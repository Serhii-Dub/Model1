let model;
let modelLoading = true;  // Змінна для перевірки, чи модель завантажена

// Завантаження моделі
async function loadModel() {
  try {
    // Завантажуємо модель ResNet50 через TensorFlow.js
    model = await tf.loadGraphModel('https://serhii-dub.github.io/Model1/resnet50_tfjs_model/model.json');
    console.log('Model Loaded');
    modelLoading = false;  // Модель завантажена
    document.getElementById('predict-button').disabled = false; // Дозволяємо передбачення
    document.getElementById('result').innerText = 'Модель готова до прогнозування!';
  } catch (error) {
    console.error('Error loading model:', error);
    document.getElementById('result').innerText = 'Помилка завантаження моделі!';
  }
}

// Завантаження зображення
document.getElementById('image-upload').addEventListener('change', function(event) {
  const file = event.target.files[0];
  if (file) {
    const reader = new FileReader();
    reader.onload = function(e) {
      const image = document.getElementById('image-preview');
      image.src = e.target.result;
      document.getElementById('predict-button').disabled = false;
    };
    reader.readAsDataURL(file);
  }
});

// Прогнозування зображення
async function predictImage() {
  if (modelLoading) {
    console.log('Waiting for the model to load...');
    document.getElementById('result').innerText = 'Модель все ще завантажується...';
    return;
  }

  if (!model) {
    console.error('Model is not loaded yet!');
    document.getElementById('result').innerText = 'Модель не завантажена!';
    return;
  }

  const imageElement = document.getElementById('image-preview');
  const imageTensor = tf.browser.fromPixels(imageElement)
    .resizeNearestNeighbor([224, 224])  // Розмір для ResNet50
    .toFloat()
    .expandDims(0)
    .div(tf.scalar(255));  // Нормалізація значень пікселів

  try {
    // Отримуємо передбачення
    const predictions = await model.predict(imageTensor);

    // Обробка результатів передбачення
    const topClass = predictions.arraySync()[0];
    const predictedLabel = topClass.indexOf(Math.max(...topClass));  // Отримуємо індекс найбільш вірогідного класу

    // Показуємо результат
    document.getElementById('result').innerText = `Передбачена категорія: ${predictedLabel}`;
  } catch (error) {
    console.error('Prediction error:', error);
    document.getElementById('result').innerText = 'Помилка при прогнозуванні!';
  }
}

// Завантажуємо модель при відкритті сторінки
window.onload = loadModel;
