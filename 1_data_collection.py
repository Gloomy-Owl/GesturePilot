import cv2
import mediapipe as mp
import os
import time
import csv

# настройки датасета
CSV_FILE = 'gestures_dataset.csv'

# создание файла
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = ['label']
        for i in range(21):
            header.extend([f'x{i}', f'y{i}', f'z{i}'])
        writer.writerow(header)

# скачиваем модель
model_path = 'hand_landmarker.task'

# настраиваем айпи
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)


def draw_landmarks(image, landmarks):
    h, w, _ = image.shape
    connections = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                   (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
                   (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
    points = []
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        points.append((cx, cy))
        cv2.circle(image, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
    for conn in connections:
        pt1, pt2 = points[conn[0]], points[conn[1]]
        cv2.line(image, pt1, pt2, (0, 255, 0), 2)


# запускаем сбор данных
detector = HandLandmarker.create_from_options(options)
cap = cv2.VideoCapture(0)

# счетчики для статистики на экране
counts = {0: 0, 1: 0, 2: 0}

print("Камера запущена. Инструкция по сбору:")
print("Нажми и УДЕРЖИВАЙ '0', чтобы записывать ЛАДОНЬ")
print("Нажми и УДЕРЖИВАЙ '1', чтобы записывать КУЛАК")
print("Нажми и УДЕРЖИВАЙ '2', чтобы записывать УКАЗАТЕЛЬНЫЙ ПАЛЕЦ")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    timestamp = int(time.time() * 1000)

    result = detector.detect_for_video(mp_image, timestamp)

    # обработка нажатий клавиш
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            draw_landmarks(frame, hand_landmarks)

            if key in [ord('0'), ord('1'), ord('2')]:
                label = chr(key)

                # координаты запястья
                wrist_x = hand_landmarks[0].x
                wrist_y = hand_landmarks[0].y
                wrist_z = hand_landmarks[0].z

                # формируем строку для CSV
                row = [label]
                for lm in hand_landmarks:
                    # вычитаем запястье, чтобы получить относительные координаты
                    row.extend([lm.x - wrist_x, lm.y - wrist_y, lm.z - wrist_z])

                # сохраняем в файл
                with open(CSV_FILE, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(row)

                counts[int(label)] += 1

    # выводим статистику на экран
    cv2.putText(frame, f"0(Palm): {counts[0]} | 1(Fist): {counts[1]} | 2(Point): {counts[2]}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('GesturePilot - Data Collection', frame)

cap.release()
cv2.destroyAllWindows()