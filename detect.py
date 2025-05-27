import cv2
import os
import numpy as np
import time
from ultralytics import YOLO
from collections import defaultdict
from typing import List, Tuple, Optional

model1 = YOLO('yolov8m.pt')  # medium - оптимально для фото по скорости и точности
model2 = YOLO('yolov8n.pt')  # nano - менее точная, но зато шустрее обработает видеофайл

# цвета для визуализации (будут использоваться и в формате RGB, и в BRG)
# синий - люди
# красный - предсказанная позиция
# зеленый - вектор движения
colors = [
    (255, 0, 0),
    (0, 0, 255),
    (0, 255, 0)
]

# параметры визуализации
BOX_THICKNESS = 4     # толщина рамки
TEXT_THICKNESS = 2    # толщина текста
TEXT_SCALE = 1.0      # размер шрифта
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_MARGIN = 10       # отступ текста от рамки

# трекинг объектов
class ObjectTracker:
    def __init__(self, max_history_length=30):  # макс. кол-во запоминаемых позиций человека
        self.track_history = defaultdict(lambda: [])
        self.max_history = max_history_length
    
    def update(self, track_id: int, position: Tuple[float, float]): # обновление истории позиций
        track = self.track_history[track_id]
        track.append(position)
        if len(track) > self.max_history:
            track.pop(0)
        return track

# предсказание будущей позиции объекта
    def predict_future_position(self, track: List[Tuple[float, float]], 
                              future_time: float, fps: float) -> Optional[Tuple[float, float]]:
        if len(track) < 2:
            return None
        
        N = min(len(track), 25) # берем последние 25 позиций объекта для расставления временных меток
        track_array = np.array(track[-N:])
        times = np.arange(-N + 1, 1)

        # линейная регрессия 
        A = np.vstack([times, np.ones(len(times))]).T
        k_x, b_x = np.linalg.lstsq(A, track_array[:, 0], rcond=None)[0]
        k_y, b_y = np.linalg.lstsq(A, track_array[:, 1], rcond=None)[0]
        
        future_frames = future_time * fps
        future_x = k_x * future_frames + b_x
        future_y = k_y * future_frames + b_y
        
        return (future_x, future_y)

# обработка входного файла: записываем время обработки; определяем, какую ф-ю детекции запускать
def detect_people(filepath):
    start_time = time.time()
    ext = os.path.splitext(filepath)[1].lower()

    if ext in ['.jpg', '.jpeg', '.png']:
        return process_image(filepath, start_time)
    elif ext in ['.mp4', '.avi', '.mov']:
        return process_video_with_tracking(filepath, start_time)
    else:
        raise ValueError("Unsupported file format")

# обработка изображений
def process_image(image_path, start_time, confidence=0.3):
    import shutil
    
    # загрузка и предобработка изображения
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # улучшение изображения для тёмных сцен
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)
    image = cv2.merge([gray, gray, gray]) if np.mean(image) < 50 else image
    
    # детекция только людей (класс 0) с заданной уверенностью
    results = model1(image, imgsz=640, conf=confidence, iou=0.4, classes=[0])[0]
    
    # результаты
    image = results.orig_img
    boxes = results.boxes
    people_boxes = boxes[boxes.cls == 0]  # Только люди
    count = len(people_boxes)
    
    # отрисовка bounding boxes для людей
    for box in people_boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        label = f"person: {conf:.2f}"
        
        cv2.rectangle(image, (x1, y1), (x2, y2), colors[1], BOX_THICKNESS)
        cv2.putText(image, label, (x1, y1 - TEXT_MARGIN),
                   TEXT_FONT, TEXT_SCALE, colors[1], TEXT_THICKNESS)
    
    # сохранение и копирование результата
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    processed_path = os.path.splitext(image_path)[0] + '_yolo.jpg'
    cv2.imwrite(processed_path, image)
    
    static_filename = os.path.basename(processed_path)
    static_path = os.path.join('static', static_filename)
    shutil.copy(processed_path, static_path)
    
    processing_time = round(time.time() - start_time, 2)
    return static_path, count, processing_time

# обработка видео с трекингом и векторами движения
def process_video_with_tracking(video_path, start_time):
    import shutil
    
    # инициализация трекера
    tracker = ObjectTracker(max_history_length=30)
    
    # открытие видео
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видеофайл: {video_path}")
    
    # получение параметров видео
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # подготовка выходного файла
    processed_path = os.path.splitext(video_path)[0] + '_yolo_track.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    writer = cv2.VideoWriter(processed_path, fourcc, fps, (width, height))
    
    frame_count = 0
    people_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # детекция и трекинг только людей
        results = model2.track(frame, persist=True, classes=[0])
        
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
    
            # визуализация базовых детекций
            annotated_frame = results[0].plot()
            
            # обработка каждого трека (все они люди, так как мы отфильтровали)
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                center = (float(x), float(y))
                
                # обновление истории трека
                track = tracker.update(track_id, center)
                
                # визуализация истории движения
                if len(track) > 1:
                    # линия трека
                    points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated_frame, [points], False, colors[2], 2)
                    
                    # вектор движения (последние 2 точки)
                    pt1 = (int(track[-2][0]), int(track[-2][1]))
                    pt2 = (int(track[-1][0]), int(track[-1][1]))
                    cv2.arrowedLine(annotated_frame, pt1, pt2, colors[2], 2, tipLength=0.5)
                    
                    # предсказание позиции через 0.5 секунды
                    future_pos = tracker.predict_future_position(track, 0.5, fps)
                    if future_pos:
                        future_x, future_y = future_pos
                        cv2.circle(annotated_frame, (int(future_x), int(future_y)), 
                                  5, colors[1], -1)
                        cv2.line(annotated_frame, (int(x), int(y)), 
                                (int(future_x), int(future_y)), colors[1], 1)
            
            # количество людей равно количеству обнаруженных коробочков
            people_count = len(boxes)
        else:
            annotated_frame = frame
            people_count = 0
        
        writer.write(annotated_frame)
        frame_count += 1
        
        # прерывание по ESC (для отладки)
        if cv2.waitKey(1) == 27:
            break
    
    # освобождение ресурсов
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    
    # копирование результата
    static_filename = os.path.basename(processed_path)
    static_path = os.path.join('static', static_filename)
    shutil.copy(processed_path, static_path)
    
    processing_time = round(time.time() - start_time, 2)
    return static_path, people_count, processing_time
