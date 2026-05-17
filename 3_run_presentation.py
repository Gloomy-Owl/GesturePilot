import sys
import cv2
import mediapipe as mp
import time
import pickle
import pyautogui
import sklearn
import os
from collections import deque

pyautogui.FAILSAFE = False

if getattr(sys, 'frozen', False):
    script_dir = os.path.dirname(sys.executable)
else:
    script_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(script_dir, 'models/gesture_model.pkl'), 'rb') as f:
    model = pickle.load(f)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=os.path.join(script_dir, 'hand_landmarker.task')),
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


BUFFER_SIZE = 4
gesture_buffer = deque(maxlen=BUFFER_SIZE)
last_action_gesture = None
current_action_text = "Ожидание..."

detector = HandLandmarker.create_from_options(options)
cap = cv2.VideoCapture(0)

print("GesturePilot Запущен!")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    timestamp = int(time.time() * 1000)

    result = detector.detect_for_video(mp_image, timestamp)

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            draw_landmarks(frame, hand_landmarks)

            row = []
            wrist = hand_landmarks[0]
            for lm in hand_landmarks:
                row.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])

            prediction = model.predict([row])[0]
            gesture_buffer.append(prediction)

            if gesture_buffer.count(prediction) == BUFFER_SIZE:

                if prediction != last_action_gesture:
                    if prediction == 1:
                        pyautogui.press('f5')
                        current_action_text = "ЛАДОНЬ -> F5 (Старт)"

                    elif prediction == 2:
                        # МАГИЯ: МЫ НИЧЕГО НЕ НАЖИМАЕМ! Это просто смена позы.
                        current_action_text = "КУЛАК -> (Нейтральное состояние)"
                        pass

                    elif prediction == 3:
                        pyautogui.press('right')
                        current_action_text = "ВЛЕВО 👈 -> Вперед / Вправо"

                    elif prediction == 4:
                        pyautogui.press('left')
                        current_action_text = "ВПРАВО 👉 -> Назад / Влево"

                    elif prediction == 5:
                        pyautogui.press('+')
                        current_action_text = "ДВА ПАЛЬЦА ✌️ -> ПРИБЛИЗИТЬ (+)"

                    elif prediction == 6:
                        pyautogui.press('up')
                        current_action_text = "ПАЛЕЦ ВВЕРХ ☝️ -> Экран Вверх"

                    elif prediction == 7:
                        pyautogui.press('down')
                        current_action_text = "ПАЛЕЦ ВНИЗ 👇 -> Экран Вниз"

                    elif prediction == 8:
                        pyautogui.press('-')
                        current_action_text = "БУКВА L 👆 -> ОТДАЛИТЬ (-)"

                    last_action_gesture = prediction

    else:
        gesture_buffer.clear()
        last_action_gesture = None
        current_action_text = "Ожидание..."

    cv2.putText(frame, current_action_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.imshow('GesturePilot', frame)

    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()