import os
from git import Repo

# Налаштування
repo_name = "image-classifier"  # Ім'я репозиторію на GitHub
github_username = "Serhii-Dub"  # Змініть на ваш логін GitHub
github_url = f"https://github.com/Serhii-Dub/Model1.git"
local_path = os.path.abspath(".")  # Поточна папка, де знаходяться ваші файли
commit_message = "Initial commit: Added Flask image classifier project"

# Список файлів для завантаження
required_files = [
    "app.py",
    "train_model.py",
    "requirements.txt",
    "templates/index.html",
    "model/model.h5",
]

# Перевірка наявності файлів
missing_files = [file for file in required_files if not os.path.exists(file)]
if missing_files:
    print("Помилка: Не знайдено файлів:")
    for file in missing_files:
        print(f"  - {file}")
    exit(1)

# Ініціалізація репозиторію
if not os.path.exists(os.path.join(local_path, ".git")):
    print("Ініціалізація нового репозиторію...")
    repo = Repo.init(local_path)

repo = Repo(local_path)

# Додавання віддаленого репозиторію
if "origin" not in [remote.name for remote in repo.remotes]:
    print("Додавання віддаленого репозиторію...")
    repo.create_remote("origin", github_url)

# Додавання всіх файлів у список
print("Додавання файлів до репозиторію...")
repo.git.add(A=True)

# Створення коміту
print("Створення коміту...")
repo.index.commit(commit_message)

# Завантаження файлів на GitHub
print("Завантаження файлів на GitHub...")
origin = repo.remotes.origin
origin.push(refspec="main:main")

print("Проект успішно завантажено на GitHub!")
