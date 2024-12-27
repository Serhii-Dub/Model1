let model;

// Завантаження моделі
async function loadModel() {
    try {
        // Завантаження моделі
        model = await tf.loadLayersModel('resnet50_tfjs_model/model.json');
        console.log('Модель успішно завантажена:', model);

        // Перевірка входу моделі і встановлення форми
        const inputShape = model.inputs[0].shape;
        if (inputShape[1] !== 224 || inputShape[2] !== 224 || inputShape[3] !== 3) {
            console.error('Невідповідний розмір вхідного зображення, очікується 224x224x3');
            document.getElementById('result').innerText = 'Невідповідний розмір зображення!';
            return;
        }

        // Виведення інформації про шари моделі
        model.layers.forEach((layer, index) => {
            console.log(`Шар ${index}:`, layer.name, layer.getConfig());
        });

        document.getElementById('result').innerText = 'Модель готова до використання!';
        document.getElementById('predict-button').disabled = false;
    } catch (error) {
        console.error('Помилка завантаження моделі:', error);
        document.getElementById('result').innerText = 'Помилка завантаження моделі!';
    }
}

// Підготовка зображення для моделі
function preprocessImage(imageElement) {
    console.log('Початкова форма зображення:', imageElement.width, imageElement.height);

    // Обробка зображення: зміна розміру та нормалізація
    const tensor = tf.browser.fromPixels(imageElement)
        .resizeNearestNeighbor([224, 224])  // Зміна розміру до 224x224
        .toFloat()  // Перетворення в float
        .div(255.0)  // Нормалізація
        .expandDims(0);  // Додавання batch dimension (обробка одного зображення)

    console.log('Форма попередньо обробленого зображення:', tensor.shape);
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
        // Прогнозування
        const predictions = model.predict(preprocessedImage);
        console.log('Сирі результати прогнозу:', predictions);

        const predictionsArray = predictions.arraySync();
        console.log('Масив прогнозів:', predictionsArray);

        // Знаходимо найбільш ймовірний клас
        const maxIndex = predictionsArray[0].indexOf(Math.max(...predictionsArray[0]));
        document.getElementById('result').innerText = `Найвірогідніше: Клас ${maxIndex} (Ймовірність: ${(predictionsArray[0][maxIndex] * 100).toFixed(2)}%)`;
    } catch (error) {
        console.error('Помилка прогнозування:', error);
        document.getElementById('result').innerText = 'Помилка прогнозування!';
    }
}

// Завантаження зображення користувачем
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
