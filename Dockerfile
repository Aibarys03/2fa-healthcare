FROM python:3.11-slim

# Устанавливаем системные зависимости для работы с изображениями
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Рабочая директория внутри контейнера
WORKDIR /app

# Сначала копируем только requirements — для кэширования
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь код проекта
COPY . .

# Создаём нужные папки
RUN mkdir -p static models data/users

# Открываем порт 8000
EXPOSE 8000

# Запускаем сервер
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
