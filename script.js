let model;

// Завантаження моделі
async function loadModel() {
  try {
    model = await tf.loadGraphModel('resnet50_tfjs_model/model.json');
    console.log('Model Loaded');
  } catch (error) {
    console.error('Error loading model:', error);
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
  if (!model) {
    console.error('Model is not loaded yet!');
    return;
  }

  const imageElement = document.getElementById('image-preview');
  const imageTensor = tf.browser.fromPixels(imageElement)
    .resizeNearestNeighbor([224, 224])  // Розмір для ResNet50
    .toFloat()
    .expandDims(0)
    .div(tf.scalar(255));

  try {
    // Отримуємо предсказання
    const predictions = await model.predict(imageTensor);
    const topClass = predictions.arraySync()[0];
    
    // Тільки для моделів ImageNet, можна замінити цей код на клас під ваші потреби
    const predictedLabel = topClass.indexOf(Math.max(...topClass));

    // Показуємо результат
    document.getElementById('result').innerText = `Predicted Class: ${predictedLabel}`;
  } catch (error) {
    console.error('Prediction error:', error);
  }
}

// Завантажуємо модель при відкритті сторінки
window.onload = loadModel;
