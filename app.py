# app.py
import os
from flask import Flask, request, render_template, redirect, url_for
from detection.detect import detect_people

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
        result_path, count, time = detect_people(filepath)

        # Передаем результат в шаблон result.html, чтобы он мог отобразить его пользователю
        return render_template('result.html',
                               result_image=os.path.basename(result_path),
                               count=count,
                               time=time)
    # Если пришел GET-запрос, то открывается форма загрузок
    return render_template('upload.html')

#@app.route('/history')
#def history():
#    records = Detection.query.order_by(Detection.timestamp.desc()).all()
#    return render_template('history.html', records=records)

# Запуск сервера в режиме разработки
if __name__ == '__main__':
    app.run(debug=True)
