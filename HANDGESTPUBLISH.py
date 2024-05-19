import cv2
import mediapipe as mp
import time
import pyautogui
import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

cv2.setUseOptimized(True)
cv2.setNumThreads(4)

def calculate_distance(x1, y1, x2, y2):
    return np.linalg.norm([x2 - x1, y2 - y1])

def init_kalman():
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([0., 0.])
    kf.F = np.array([[1., 1.], [0., 1.]])
    kf.H = np.array([[1., 0.]])
    kf.P *= 500.
    kf.R = 2  # Reduce measurement noise
    kf.Q = Q_discrete_white_noise(dim=2, dt=1., var=0.01)  # Reduce process noise
    return kf

kf_x = init_kalman()
kf_y = init_kalman()

cap = cv2.VideoCapture(0)
cap.set(3, 260)  # Reduce resolution width
cap.set(4, 220)  # Reduce resolution height

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

pTime = 0
lastClickTime = 0
x_movement_threshold = 2  # Lower the threshold for more sensitivity
y_movement_threshold = 2  # Lower the threshold for more sensitivity
x_sensitivity_multiplier = 1.0
y_sensitivity_multiplier = 1.0

prev_x, prev_y = 0, 0

screenWidth, screenHeight = pyautogui.size()
screenWidth *= 2

pyautogui.FAILSAFE = False

alpha = 0.5  # Adjust EMA smoothing factor for better balance
ema_x, ema_y = 0, 0
initialized = False

while True:
    success, img = cap.read()
    if not success or img is None:
        print("Failed to capture image")
        continue

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            h, w, c = img.shape
            cx, cy = 0, 0  # Initialize cx, cy
            for id, lm in enumerate(handLms.landmark):
                if id == 0:  # Wrist landmark
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    dx = abs(cx - prev_x)
                    dy = abs(cy - prev_y)

                    if dx > x_movement_threshold or dy > y_movement_threshold:
                        cx_amplified = (cx - prev_x) * x_sensitivity_multiplier + prev_x
                        cy_amplified = (cy - prev_y) * y_sensitivity_multiplier + prev_y

                        kf_x.predict()
                        kf_y.predict()

                        kf_x.update([cx])
                        kf_y.update([cy])

                        filtered_x = kf_x.x[0]
                        filtered_y = kf_y.x[0]

                        if not initialized:
                            ema_x, ema_y = filtered_x, filtered_y
                            initialized = True
                        else:
                            ema_x = alpha * filtered_x + (1 - alpha) * ema_x
                            ema_y = alpha * filtered_y + (1 - alpha) * ema_y

                        screenX = np.interp(ema_x, (0, w), (0, screenWidth))
                        screenY = np.interp(ema_y, (0, h), (0, screenHeight))
                        pyautogui.moveTo(screenX, screenY)

                    prev_x, prev_y = cx, cy

            if len(handLms.landmark) > 8:
                x4, y4 = int(handLms.landmark[4].x * w), int(handLms.landmark[4].y * h)
                x8, y8 = int(handLms.landmark[8].x * w), int(handLms.landmark[8].y * h)
                distance = calculate_distance(x4, y4, x8, y8)
                if distance < 25 and (time.time() - lastClickTime) > 0.7:
                    pyautogui.click()
                    lastClickTime = time.time()

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

