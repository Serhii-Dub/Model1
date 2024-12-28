async function loadModel() {
    try {
        const modelUrl = 'https://serhii-dub.github.io/Model1/resnet50_tfjs_model/model.json';
        model = await tf.loadLayersModel(modelUrl);

        // Додавання вхідного шару, якщо його немає
        if (!model.inputShape) {
            const inputShape = [224, 224, 3]; // RGB
            const newInputLayer = tf.input({ shape: inputShape });
            const newOutputLayer = model.apply(newInputLayer);
            model = tf.model({ inputs: newInputLayer, outputs: newOutputLayer });
        }

        console.log("Модель успішно завантажена!");
    } catch (error) {
        console.error('Помилка завантаження моделі:', error);
    }
}
async function predict() {
    const imgElement = document.getElementById('input-image');
    const resultElement = document.getElementById('result');

    if (!model) {
        console.error("Модель ще не завантажена!");
        resultElement.textContent = "Модель ще не завантажена!";
        return;
    }

    if (!imgElement.complete || imgElement.naturalWidth === 0) {
        console.error('Зображення не завантажене належним чином!');
        resultElement.textContent = 'Зображення не завантажене належним чином!';
        return;
    }

    // Попередня обробка зображення
    const processedImage = tf.browser.fromPixels(imgElement)
        .resizeNearestNeighbor([224, 224])
        .toFloat()
        .expandDims(0);

    try {
        const prediction = await model.predict(processedImage).data();
        console.log('Прогноз:', prediction);

        const predictedClass = Array.from(prediction).map((value, index) => ({ index, value }))
            .sort((a, b) => b.value - a.value)[0];
        resultElement.textContent = `Найімовірніший клас: ${predictedClass.index}, Значення: ${predictedClass.value.toFixed(4)}`;
    } catch (error) {
        console.error('Помилка при прогнозуванні:', error);
        resultElement.textContent = 'Помилка при прогнозуванні.';
    }
}
