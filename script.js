let model;
let modelLoading = true;

// Завантаження моделі
async function loadModel() {
    try {
        // Завантажуємо модель без модифікацій
        model = await tf.loadLayersModel('https://serhii-dub.github.io/Model1/resnet50_tfjs_model/model.json');
        console.log('Модель успішно завантажена');
        modelLoading = false;
        document.getElementById('predict-button').disabled = false;
        document.getElementById('result').innerText = 'Модель готова до прогнозування!';
    } catch (error) {
        console.error('Помилка завантаження моделі:', error);
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
        document.getElementById('result').innerText = 'Модель все ще завантажується...';
        return;
    }

    if (!model) {
        console.error('Модель не завантажена!');
        document.getElementById('result').innerText = 'Модель не завантажена!';
        return;
    }

    const imageElement = document.getElementById('image-preview');
    const imageTensor = tf.browser.fromPixels(imageElement)
        .resizeNearestNeighbor([224, 224]) // Розмір для ResNet50
        .toFloat()
        .expandDims(0)
        .div(tf.scalar(255)); // Нормалізація

    try {
        // Отримуємо передбачення
        const predictions = await model.predict(imageTensor).array();

        // Обробляємо передбачення
        const predictedIndex = predictions[0].indexOf(Math.max(...predictions[0]));

        document.getElementById('result').innerText = `Передбачена категорія: ${predictedIndex}`;
    } catch (error) {
        console.error('Помилка передбачення:', error);
        document.getElementById('result').innerText = 'Помилка передбачення!';
    }
}

// Завантаження моделі при старті сторінки
window.onload = loadModel;
