// Завантаження моделі
async function loadModel() {
    try {
        // Завантажуємо модель ResNet50
        console.log("Завантаження моделі...");
        const model = await tf.loadLayersModel('resnet50_tfjs_model/model.json');

        // Перевірка форми входу і налаштування
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
