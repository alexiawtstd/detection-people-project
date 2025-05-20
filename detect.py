import cv2
import os
import numpy as np
import time
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

colors = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
    (255, 0, 255), (192, 192, 192), (128, 128, 128), (128, 0, 0), (128, 128, 0),
    (0, 128, 0), (128, 0, 128), (0, 128, 128), (0, 0, 128), (72, 61, 139),
    (47, 79, 79), (47, 79, 47), (0, 206, 209), (148, 0, 211), (255, 20, 147)
]

def detect_people(filepath):
    start_time = time.time()
    ext = os.path.splitext(filepath)[1].lower()

    if ext in ['.jpg', '.jpeg', '.png']:
        return process_image(filepath, start_time)
    elif ext in ['.mp4', '.avi', '.mov']:
        return process_video(filepath, start_time)
    else:
        raise ValueError("Unsupported file format")


def process_image(image_path, start_time):
    import shutil

    image = cv2.imread(image_path)
    results = model(image)[0]

    image = results.orig_img
    class_ids = results.boxes.cls.cpu().numpy()
    boxes = results.boxes.xyxy.cpu().numpy().astype(np.int32)
    names = results.names

    count = 0
    for class_id, box in zip(class_ids, boxes):
        if int(class_id) == 0:
            count += 1
            color = colors[int(class_id) % len(colors)]
            x1, y1, x2, y2 = box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, 'person', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Сохраняем временно в uploads
    processed_path = os.path.splitext(image_path)[0] + '_yolo.jpg'
    cv2.imwrite(processed_path, image)

    # Копируем в static
    static_filename = os.path.basename(processed_path)
    static_path = os.path.join('static', static_filename)
    shutil.copy(processed_path, static_path)

    processing_time = round(time.time() - start_time, 2)
    return static_path, count, processing_time


def process_video(video_path, start_time):
    import shutil

    capture = cv2.VideoCapture(video_path)
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    processed_path = os.path.splitext(video_path)[0] + '_yolo.mp4'
    # Используем более совместимый кодек
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # или 'h264'
    writer = cv2.VideoWriter(processed_path, fourcc, fps, (width, height))

    total_person_count = 0
    frame_count = 0

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        results = model(frame)[0]
        class_ids = results.boxes.cls.cpu().numpy()
        boxes = results.boxes.xyxy.cpu().numpy().astype(np.int32)

        person_count = 0
        for class_id, box in zip(class_ids, boxes):
            if int(class_id) == 0:  # 0 - это класс 'person' в YOLO
                person_count += 1
                color = colors[int(class_id) % len(colors)]
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, 'person', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        total_person_count += person_count
        frame_count += 1
        writer.write(frame)

    capture.release()
    writer.release()

    # Если видео пустое, все равно создаем файл
    if frame_count == 0:
        # Создаем черный кадр
        blank_frame = np.zeros((height, width, 3), dtype=np.uint8)
        for _ in range(int(fps)):  # 1 секунда видео
            writer.write(blank_frame)
        writer.release()

    # Копируем в static/
    static_filename = os.path.basename(processed_path)
    static_path = os.path.join('static', static_filename)
    shutil.copy(processed_path, static_path)

    processing_time = round(time.time() - start_time, 2)
    return static_path, total_person_count, processing_time
