import os
from git import Repo

# Налаштування
repo_name = "Model1_IM-24"  # Ім'я репозиторію на GitHub
github_username = "Serhii-Dub"  # Змініть на ваш логін GitHub
github_url = "https://github.com/Serhii-Dub/Model1_IM-24.git"  # Правильний URL репозиторію
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
else:
    repo = Repo(local_path)

# Створення README.md файлу, якщо його немає
readme_path = os.path.join(local_path, "README.md")
if not os.path.exists(readme_path):
    print("Створення README.md файлу...")
    with open(readme_path, "w") as readme_file:
        readme_file.write(f"# {repo_name}\n")

# Переведення на гілку main (якщо її ще немає)
print("Переведення на гілку main...")
repo.git.branch("-M", "main")

# Додавання віддаленого репозиторію
print("Додавання віддаленого репозиторію...")
if "origin" not in [remote.name for remote in repo.remotes]:
    repo.create_remote("origin", github_url)

# Додавання всіх файлів до репозиторію
print("Додавання файлів до репозиторію...")
repo.git.add(A=True)

# Перевірка наявності комітів і створення першого коміту, якщо репозиторій порожній
if not repo.head.is_valid():
    print("Ініціалізація репозиторію: створення початкового коміту...")
    repo.index.commit(commit_message)

# Створення коміту
print("Створення коміту...")
repo.index.commit(commit_message)

# Завантаження файлів на GitHub
print("Завантаження файлів на GitHub...")
origin = repo.remotes.origin

# Перевірка статусу гілки
print("Перевірка статусу гілки...")
status = repo.git.status()
print("Status:\n", status)

# Пуш на GitHub
try:
    origin.push(refspec="main:main")
    print("Проект успішно завантажено на GitHub!")
except Exception as e:
    print("Помилка при завантаженні на GitHub:", e)
