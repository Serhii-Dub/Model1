// Завантаження моделі
async function loadModel() {
    try {
        model = await tf.loadLayersModel('https://serhii-dub.github.io/Model1/resnet50_tfjs_model/model.json');
        console.log("Модель завантажена!");

        // Перевірка вхідних даних
        const inputLayer = model.input;
        console.log('Модель має вхідний шар:', inputLayer);

        // Якщо модель не має визначеного вхідного шару, намагаємось додати його вручну
        if (!inputLayer) {
            console.error("Модель не має визначеного вхідного шару.");
        }

        // Якщо вхідний шар є, можна продовжити з прогнозами
        const inputShape = model.inputShape || [null, null, 3];  // припускаємо, що модель має 3 канали (RGB)
        console.log('Input Shape:', inputShape);

    } catch (error) {
        console.error('Помилка завантаження моделі:', error);
    }
}

// Викликаємо функцію завантаження
loadModel();


async function predict() {
    const imageElement = document.getElementById('input-image'); // Ваше зображення
    if (!imageElement) {
        console.error('Зображення не знайдено!');
        return;
    }
    
    // Попередня обробка зображення: зміна розміру до 224x224 і нормалізація
    const processedImage = tf.browser.fromPixels(imageElement)
        .resizeNearestNeighbor([224, 224])  // змінюємо розмір до 224x224
        .toFloat()                         // перетворюємо на float
        .expandDims(0);                     // додаємо вимір для пакету (batch size)

    // Прогнозування
    try {
        const prediction = await model.predict(processedImage);
        prediction.print(); // Виведення результату прогнозу в консоль
    } catch (error) {
        console.error('Помилка при прогнозуванні:', error);
    }
}
