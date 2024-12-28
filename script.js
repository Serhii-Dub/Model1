let model;

// Завантаження моделі
async function loadModel() {
    try {
        const modelUrl = 'https://serhii-dub.github.io/Model1/resnet50_tfjs_model/model.json';
        model = await tf.loadLayersModel(modelUrl);

        // Перевірка вхідного шару
        if (!model.inputShape) {
            const inputShape = [224, 224, 3];
            const newInputLayer = tf.input({ shape: inputShape });
            const newOutputLayer = model.apply(newInputLayer);
            model = tf.model({ inputs: newInputLayer, outputs: newOutputLayer });
        }

        console.log("Модель успішно завантажена!");
    } catch (error) {
        console.error('Помилка завантаження моделі:', error);
    }
}

// Виклик функції завантаження моделі
loadModel();

// Додавання слухача до завантаження зображення
document.getElementById('file-input').addEventListener('change', (event) => {
    const file = event.target.files[0];
    const imgElement = document.getElementById('input-image');
    const predictButton = document.getElementById('predict-button');

    if (file) {
        const reader = new FileReader();
        reader.onload = () => {
            imgElement.src = reader.result;
            imgElement.style.display = 'block';
            predictButton.disabled = false; // Активувати кнопку
        };
        reader.readAsDataURL(file);
    }
});

// Функція прогнозування
async function predict() {
    const imgElement = document.getElementById('input-image');
    const resultElement = document.getElementById('result');

    if (!imgElement.complete || imgElement.naturalWidth === 0) {
        console.error('Зображення не завантажене належним чином!');
        return;
    }

    // Попередня обробка зображення
    const processedImage = tf.browser.fromPixels(imgElement)
        .resizeNearestNeighbor([224, 224])  // Змінюємо розмір до 224x224
        .toFloat()
        .expandDims(0); // Додаємо вимір для batch

    try {
        const prediction = await model.predict(processedImage).data();
        console.log('Прогноз:', prediction);

        // Виведення результату
        const predictedClass = Array.from(prediction).map((value, index) => ({ index, value }))
            .sort((a, b) => b.value - a.value)[0];
        resultElement.textContent = `Найімовірніший клас: ${predictedClass.index}, Значення: ${predictedClass.value.toFixed(4)}`;
    } catch (error) {
        console.error('Помилка при прогнозуванні:', error);
        resultElement.textContent = 'Помилка при прогнозуванні.';
    }
}
