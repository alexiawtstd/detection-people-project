# detection/detect.py
import time
import shutil
import os

# Начало функции. Она получает путь к загруженному файлу
def detect_people(filepath):
    start_time = time.time() # Запоминаем время начала обработки

    # Просто копируем файл в static как "обработанный" с новым именем. Пока что это имитирует обработку
    filename = os.path.basename(filepath)
    processed_path = os.path.join('static', f'processed_{filename}')
    shutil.copy(filepath, processed_path)

    # Фейковые результаты (3 человека и время обработки в секундах). После работы с YOLO будут уже настоящие данные
    people_count = 3
    processing_time = round(time.time() - start_time, 2)

    # Возвращаем результат обратно в app.py
    return processed_path, people_count, processing_time
