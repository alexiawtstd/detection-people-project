# app.py
import os
from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from detection.detect import detect_people
from database import Session, MediaFile



# Создаем Flask-приложение
app = Flask(__name__)

# Папки для сохранения загрузок
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Убеждаемся, что папки существуют. Если нет, то создаем их
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# Главная страница сайта. GET - открытие страницы, POST - отправка формы с файлом
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST': # Когда пользователь нажимает "загрузить", приходит POST-запрос
        file = request.files['file'] # Получаем файл. Если пусто, то ошибка 400
        if not file:
            return "Файл не выбран", 400

        # Находим путь и сохраняем файл
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Обработка файла — вернёт путь, число людей и время обработки.
        # Все эти данные приходят из функции detect_people(), которая определена в detect.py
        processed_path, people_count, processing_time = detect_people(filepath)

        # Определяем тип файла
        file_ext = os.path.splitext(file.filename)[1].lower()
        file_type = 'photo' if file_ext in ['.jpg', '.jpeg', '.png'] else 'video'


        session = Session()
        media = MediaFile(
            filename=file.filename,
            original_filepath=os.path.basename(filepath),
            filepath=os.path.basename(processed_path),
            file_type=file_type,
            processing_time=processing_time,
            people_count=people_count if file_type == 'photo' else None,
            width=None,
            height=None,
            duration=None,
            frame_count=None
        )

        session.add(media)
        session.commit()
        session.close()

        # Передаем результат в шаблон result.html, чтобы он мог отобразить его пользователю
        return render_template('result.html',
                               processed_path=os.path.basename(processed_path),
                               people_count=people_count,
                               processing_time=processing_time)
    # Если пришел GET-запрос, то открывается форма загрузок
    return render_template('upload.html')

@app.route('/history')
def history():
    session = Session()
    records = session.query(MediaFile).order_by(MediaFile.upload_time.desc()).all()
    session.close()
    return render_template('history.html', records=records)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# Запуск сервера в режиме разработки
if __name__ == '__main__':
    app.run(debug=True)
