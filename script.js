let model;

// Завантаження моделі
async function loadModel() {
    try {
        // Завантажуємо модель
        model = await tf.loadLayersModel('resnet50_tfjs_model/model.json');
        console.log('Модель успішно завантажена:', model);

        // Перевірка конфігурації першого шару
        const inputLayer = model.layers[0];
        console.log('Конфігурація першого шару:', inputLayer.getConfig());

        // Вивід вхідної форми
        console.log('Вхідна форма моделі:', model.inputs[0].shape);

        document.getElementById('result').innerText = 'Модель готова до використання!';
        document.getElementById('predict-button').disabled = false;
    } catch (error) {
        console.error('Помилка завантаження моделі:', error);
        document.getElementById('result').innerText = 'Помилка завантаження моделі!';
    }
}

// Підготовка зображення
function preprocessImage(imageElement) {
    console.log('Початкова форма зображення:', imageElement.width, imageElement.height);

    const tensor = tf.browser.fromPixels(imageElement)
        .resizeNearestNeighbor([224, 224]) // Зміна розміру зображення
        .toFloat()
        .div(255.0) // Нормалізація до [0, 1]
        .expandDims(0); // Додаємо batch dimension

    console.log('Форма попередньо обробленого тензора:', tensor.shape);
    return tensor;
}

// Прогнозування
async function predictImage() {
    if (!model) {
        document.getElementById('result').innerText = 'Модель не завантажена!';
        return;
    }

    const imageElement = document.getElementById('image-preview');
    const preprocessedImage = preprocessImage(imageElement);

    try {
        const predictions = model.predict(preprocessedImage);
        console.log('Сирі результати прогнозу:', predictions);

        const predictionsArray = predictions.arraySync();
        console.log('Масив прогнозів:', predictionsArray);

        const maxIndex = predictionsArray[0].indexOf(Math.max(...predictionsArray[0]));
        document.getElementById('result').innerText = `Найвірогідніше: Клас ${maxIndex} (Ймовірність: ${(predictionsArray[0][maxIndex] * 100).toFixed(2)}%)`;
    } catch (error) {
        console.error('Помилка прогнозування:', error);
        document.getElementById('result').innerText = 'Помилка прогнозування!';
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
        };
        reader.readAsDataURL(file);
    }
});

// Завантаження моделі при запуску сторінки
window.onload = loadModel;
