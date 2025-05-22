import os
from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from detection.detect import detect_people
from database import Session, MediaFile
import cv2


app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if not file:
            return "Файл не выбран", 400

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        processed_path, people_count, processing_time = detect_people(filepath)

        file_ext = os.path.splitext(file.filename)[1].lower()
        file_type = 'photo' if file_ext in ['.jpg', '.jpeg', '.png'] else 'video'

        file_size = os.path.getsize(filepath)

        duration = None
        frame_count = None

        if file_type == 'video':
            cap = cv2.VideoCapture(filepath)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else None
            cap.release()


        session = Session()
        media = MediaFile(
            filename=file.filename,
            original_filepath=os.path.basename(filepath),
            filepath=os.path.basename(processed_path),
            file_type=file_type,
            processing_time=processing_time,
            people_count=people_count if file_type == 'photo' else None,
            duration = duration,
            frame_count = frame_count,
            status='processed'
        )

        media.file_size = file_size

        session.add(media)
        session.commit()
        session.close()

        return render_template('result.html',
                               processed_path=os.path.basename(processed_path),
                               people_count=people_count,
                               processing_time=processing_time)
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


if __name__ == '__main__':
    app.run(debug=True)
