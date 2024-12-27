let model;
let modelLoading = true;  // Змінна для перевірки, чи модель завантажена

// Завантаження моделі
async function loadModel() {
  try {
    model = await tf.loadGraphModel('resnet50_tfjs_model/model.json');
    console.log('Model Loaded');
    modelLoading = false;  // Модель завантажена
    document.getElementById('predict-button').disabled = false; // Дозволяємо передбачення
    document.getElementById('result').innerText = 'Model is ready to predict!';
  } catch (error) {
    console.error('Error loading model:', error);
    document.getElementById('result').innerText = 'Error loading model!';
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
    document.getElementById('result').innerText = 'Model is still loading...';
    return;
  }

  if (!model) {
    console.error('Model is not loaded yet!');
    document.getElementById('result').innerText = 'Model failed to load!';
    return;
  }

  const imageElement = document.getElementById('image-preview');
  const imageTensor = tf.browser.fromPixels(imageElement)
    .resizeNearestNeighbor([224, 224])  // Розмір для ResNet50
    .toFloat()
    .expandDims(0)
    .div(tf.scalar(255));

  try {
    // Отримуємо передбачення
    const predictions = await model.predict(imageTensor);
    const topClass = predictions.arraySync()[0];
    
    // Тільки для моделей ImageNet, можна замінити цей код на ваші класи
    const predictedLabel = topClass.indexOf(Math.max(...topClass));

    // Показуємо результат
    document.getElementById('result').innerText = `Predicted Class: ${predictedLabel}`;
  } catch (error) {
    console.error('Prediction error:', error);
    document.getElementById('result').innerText = 'Prediction failed!';
  }
}

// Завантажуємо модель при відкритті сторінки
window.onload = loadModel;
