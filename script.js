async function loadModel() {
    try {
        // Завантаження моделі
        console.log("Завантаження моделі...");
        let model = await tf.loadLayersModel('resnet50_tfjs_model/model.json');

        // Перевірка, чи є inputShape або batchInputShape в моделі
        if (!model.inputs || !model.inputs[0].shape) {
            // Якщо форма входу не визначена, явно задаємо форму входу
            console.log("Форма входу не вказана, задаємо її вручну...");
            model.build([null, 224, 224, 3]); // Встановлюємо вхідну форму для ResNet50
        }

        console.log('Модель успішно завантажена:', model);

        // Виведення інформації про шари моделі
        model.layers.forEach((layer, index) => {
            console.log(`Шар ${index}:`, layer.name, layer.getConfig());
        });

        // Виведення вхідної форми
        console.log('Вхідна форма моделі:', model.inputs[0].shape);

        // Оновлення тексту на сторінці
        document.getElementById('result').innerText = 'Модель готова до використання!';
        document.getElementById('predict-button').disabled = false;
    } catch (error) {
        console.error('Помилка завантаження моделі:', error);
        document.getElementById('result').innerText = 'Помилка завантаження моделі!';
    }
}

// Викликаємо функцію завантаження моделі при завантаженні сторінки
window.onload = loadModel;
