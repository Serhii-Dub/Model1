let model;

// Завантаження моделі ResNet
async function loadModel() {
  model = await tf.loadGraphModel('model.json');
  console.log('Model Loaded');
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
  const imageElement = document.getElementById('image-preview');
  const imageTensor = tf.browser.fromPixels(imageElement)
    .resizeNearestNeighbor([224, 224])  // Розмір для ResNet
    .toFloat()
    .expandDims(0)
    .div(tf.scalar(255));

  const predictions = await model.predict(imageTensor);
  const topClass = predictions.arraySync()[0];
  const predictedLabel = topClass.indexOf(Math.max(...topClass)); // Індекс найбільш імовірного класу

  document.getElementById('result').innerText = `Predicted Class: ${predictedLabel}`;
}

// Завантажуємо модель при відкритті сторінки
window.onload = loadModel;
