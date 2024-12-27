let model;

// Завантаження моделі MobileNet
async function loadModel() {
    try {
        model = await mobilenet.load();
        console.log('Модель успішно завантажена');
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

// Прогнозування
document.getElementById('predict-button').addEventListener('click', async function() {
    const imageElement = document.getElementById('image-preview');
    
    if (!model) {
        document.getElementById('result').innerText = 'Модель не завантажена!';
        return;
    }

    try {
        const predictions = await model.classify(imageElement);
        console.log('Результати прогнозу:', predictions);
        document.getElementById('result').innerText = `Найвірогідніше: ${predictions[0].className} (${(predictions[0].probability * 100).toFixed(2)}%)`;
    } catch (error) {
        console.error('Помилка прогнозування:', error);
        document.getElementById('result').innerText = 'Помилка прогнозування!';
    }
});

// Завантажуємо модель при завантаженні сторінки
window.onload = loadModel;
