import cv2
import mediapipe as mp
import os
import time
import csv
import urllib.request

script_dir = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(script_dir, 'gestures_dataset.csv')
model_path = os.path.join(script_dir, 'hand_landmarker.task')

if not os.path.exists(model_path):
    print("Скачиваю модель...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        model_path)

if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = ['label']
        for i in range(21):
            header.extend([f'x{i}', f'y{i}', f'z{i}'])
        writer.writerow(header)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
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


detector = HandLandmarker.create_from_options(options)
cap = cv2.VideoCapture(0)

counts = {i: 0 for i in range(1, 9)}  # Теперь 8 жестов

print("=== СБОР 8 ЖЕСТОВ ===")
print("1 - Ладонь 🖐️ (Старт)")
print("2 - Кулак ✊ (ХОЛОСТОЙ ХОД / ПАУЗА МЕЖДУ ЖЕСТАМИ)")
print("3 - Большой палец ВЛЕВО 👈 (Вперед / Вправо)")
print("4 - Большой палец ВПРАВО 👉 (Назад / Влево)")
print("5 - Два пальца ✌️ (Приблизить)")
print("6 - Палец ВВЕРХ ☝️ (Экран Вверх)")
print("7 - Палец ВНИЗ 👇 (Экран Вниз)")
print("8 - Буква L 👆👈 (Отдалить обратно)")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    timestamp = int(time.time() * 1000)

    result = detector.detect_for_video(mp_image, timestamp)
    key = cv2.waitKey(1) & 0xFF
    if key == 27: break

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            draw_landmarks(frame, hand_landmarks)

            # Слушаем кнопки от 1 до 8
            if key in [ord(str(i)) for i in range(1, 9)]:
                label = chr(key)
                wrist = hand_landmarks[0]
                row = [label]
                for lm in hand_landmarks:
                    row.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])

                with open(CSV_FILE, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(row)

                counts[int(label)] += 1

    stats1 = f"1:{counts[1]} 2:{counts[2]} 3:{counts[3]} 4:{counts[4]}"
    stats2 = f"5:{counts[5]} 6:{counts[6]} 7:{counts[7]} 8:{counts[8]}"
    cv2.putText(frame, stats1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(frame, stats2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow('Data Collection', frame)

cap.release()
cv2.destroyAllWindows()