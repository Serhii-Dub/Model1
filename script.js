// Завантаження моделі
async function loadModel() {
    try {
        // Завантажуємо модель ResNet50
        console.log("Завантаження моделі...");
        const model = await tf.loadLayersModel('resnet50_tfjs_model/model.json');

        // Перевіряємо форму входу
        if (!model.inputs || !model.inputs[0].shape) {
            console.log("Форма входу не вказана, задаємо її вручну...");
            model.build([null, 224, 224, 3]); // Встановлюємо вхідну форму для ResNet50
        }

        console.log('Модель успішно завантажена:', model);

        // Виводимо інформацію про шари моделі
        model.layers.forEach((layer, index) => {
            console.log(`Шар ${index}:`, layer.name, layer.getConfig());
        });

        // Виводимо вхідну форму
        console.log('Вхідна форма моделі:', model.inputs[0].shape);

        // Оновлення тексту на сторінці
        document.getElementById('result').innerText = 'Модель готова до використання!';
        document.getElementById('predict-button').disabled = false;
    } catch (error) {
        console.error('Помилка завантаження моделі:', error);
        document.getElementById('result').innerText = 'Помилка завантаження моделі!';
    }
}

// Завантажуємо модель при завантаженні сторінки
window.onload = loadModel;

// Функція для прогнозування
async function predictImage() {
    const fileInput = document.getElementById('image-upload');
    const file = fileInput.files[0];
    
    if (!file) {
        alert("Будь ласка, завантажте зображення!");
        return;
    }

    // Відображаємо попередній перегляд зображення
    const imgElement = document.getElementById('image-preview');
    imgElement.src = URL.createObjectURL(file);

    // Завантажуємо зображення та змінюємо його розмір
    const img = await loadImage(file);
    const tensor = tf.browser.fromPixels(img).resizeNearestNeighbor([224, 224]).toFloat().expandDims(0);

    // Нормалізація пікселів зображення (якщо потрібно)
    const normalized = tensor.div(tf.scalar(255));

    // Робимо прогноз
    const model = await tf.loadLayersModel('resnet50_tfjs_model/model.json');
    const predictions = await model.predict(normalized);

    // Виводимо результат
    const resultDiv = document.getElementById('result');
    resultDiv.innerHTML = `Прогноз: ${predictions}`;
}

// Завантаження зображення для прогнозу
function loadImage(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = new Image();
            img.onload = () => resolve(img);
            img.onerror = (err) => reject(err);
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);
    });
}
