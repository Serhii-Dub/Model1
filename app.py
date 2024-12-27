import os
import torch
import torchvision
import torchvision.transforms as transforms
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import io

# Налаштування
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Завантаження попередньо навченого ResNet
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torchvision.models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, 1000)  # Стандартна кількість класів ImageNet
model.load_state_dict(torch.load('resnet_model.pth'))
model.eval().to(device)

# Трансформація зображень
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Перевірка розширення файлу
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Обробка зображень і прогнозування
def predict_image(image):
    image = Image.open(io.BytesIO(image))
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    return predicted.item()

# Головна сторінка
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Прогнозування
            with open(filepath, 'rb') as f:
                img_data = f.read()
                prediction = predict_image(img_data)

            return render_template('index.html', prediction=prediction, image_url=filepath)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
