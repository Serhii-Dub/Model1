let model;

// Завантаження моделі
async function loadModel() {
    try {
        model = await tf.loadLayersModel('resnet50_tfjs_model/model.json');
        console.log('Модель успішно завантажена:', model);

        model.layers.forEach((layer, index) => {
            console.log(`Шар ${index}:`, layer.name, layer.getConfig());
        });

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
    const tensor = tf.browser.fromPixels(imageElement)
        .resizeNearestNeighbor([224, 224])
        .toFloat()
        .div(255.0)
        .expandDims(0);
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

    document.getElementById('result').innerText = 'Прогнозування триває...';

    try {
        const predictions = model.predict(preprocessedImage);
        const predictionsArray = predictions.arraySync();
        const topN = 3;

        const sortedPredictions = predictionsArray[0]
            .map((probability, index) => ({ index, probability }))
            .sort((a, b) => b.probability - a.probability)
            .slice(0, topN);

        const resultText = sortedPredictions.map(
            (pred, rank) => `#${rank + 1}: Клас ${pred.index} (Ймовірність: ${(pred.probability * 100).toFixed(2)}%)`
        ).join('\n');

        document.getElementById('result').innerText = `Результати прогнозу:\n${resultText}`;
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
    } else {
        document.getElementById('result').innerText = 'Будь ласка, виберіть файл зображення.';
    }
});

// Завантаження моделі при запуску сторінки
window.onload = loadModel;
