import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import pyttsx3
from collections import deque
import time

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)
voices = engine.getProperty('voices')
if len(voices) > 1:
    engine.setProperty('voice', voices[1].id)  # Female voice if available

# Voice feedback management
last_voice_time = 0
voice_cooldown = 1.0  # Minimum seconds between voice feedback


def speak(text):
    global last_voice_time
    current_time = time.time()
    if current_time - last_voice_time > voice_cooldown:
        engine.say(text)
        engine.runAndWait()
        last_voice_time = current_time


# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# Screen dimensions
screen_w, screen_h = pyautogui.size()

# Mouse movement smoothing
smoothening = 7
mouse_points = deque(maxlen=smoothening)

# Gesture state tracking
dragging = False
last_gesture = None
gesture_cooldown = 0.5  # Seconds to wait before repeating the same gesture

# Video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video capture")
    exit()


# Gesture detection functions
def is_pinch(thumb, index, threshold=0.03):
    distance = ((thumb.x - index.x) ** 2 + (thumb.y - index.y) ** 2) ** 0.5
    return distance < threshold


def is_right_click(thumb, middle, threshold=0.04):
    distance = ((thumb.x - middle.x) ** 2 + (thumb.y - middle.y) ** 2) ** 0.5
    return distance < threshold


def is_fist(landmarks):
    palm = landmarks.landmark[0]
    fingers_down = 0

    for tip_id in [8, 12, 16, 20]:  # Tips of index, middle, ring, pinky
        tip = landmarks.landmark[tip_id]
        distance = ((tip.x - palm.x) ** 2 + (tip.y - palm.y) ** 2) ** 0.5
        if distance < 0.1:
            fingers_down += 1

    return fingers_down >= 3  # At least 3 fingers closed


def is_scroll_gesture(landmarks):
    index_tip = landmarks.landmark[8]
    middle_tip = landmarks.landmark[12]
    index_base = landmarks.landmark[5].y
    middle_base = landmarks.landmark[9].y

    return (index_tip.y < index_base and middle_tip.y < middle_base)


def is_double_click_gesture(thumb, index, middle, threshold=0.05):
    # Thumb near both index and middle
    dist1 = ((thumb.x - index.x) ** 2 + (thumb.y - index.y) ** 2) ** 0.5
    dist2 = ((thumb.x - middle.x) ** 2 + (thumb.y - middle.y) ** 2) ** 0.5
    return dist1 < threshold and dist2 < threshold


def is_zoom_gesture(landmarks):
    # Check for spread (zoom in) or pinch (zoom out) between index and thumb
    thumb_tip = landmarks.landmark[4]
    index_tip = landmarks.landmark[8]
    distance = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5
    return distance > 0.15  # Threshold for zoom gesture


# Main loop
prev_time = time.time()
while True:
    success, img = cap.read()
    if not success:
        continue

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    current_gesture = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get landmarks
            thumb = hand_landmarks.landmark[4]
            index = hand_landmarks.landmark[8]
            middle = hand_landmarks.landmark[12]
            ring = hand_landmarks.landmark[16]
            pinky = hand_landmarks.landmark[20]
            palm = hand_landmarks.landmark[0]

            # Convert index finger position to screen coords
            curr_x = int(index.x * screen_w)
            curr_y = int(index.y * screen_h)

            # Smoothing (moving average)
            mouse_points.append((curr_x, curr_y))
            avg_x = int(np.mean([x for x, y in mouse_points]))
            avg_y = int(np.mean([y for x, y in mouse_points]))

            # Move mouse
            pyautogui.moveTo(avg_x, avg_y)

            # Gesture detection with priority
            if is_double_click_gesture(thumb, index, middle):
                current_gesture = "double_click"
                cv2.putText(img, "DOUBLE CLICK", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 200), 2)
                if last_gesture != "double_click":
                    pyautogui.doubleClick()
                    speak("Double click")

            elif is_pinch(thumb, index):
                current_gesture = "left_click"
                cv2.putText(img, "LEFT CLICK", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if last_gesture != "left_click":
                    pyautogui.click()
                    speak("Click")

            elif is_right_click(thumb, middle):
                current_gesture = "right_click"
                cv2.putText(img, "RIGHT CLICK", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if last_gesture != "right_click":
                    pyautogui.rightClick()
                    speak("Right click")

            elif is_fist(hand_landmarks):
                current_gesture = "drag"
                if not dragging:
                    pyautogui.mouseDown()
                    dragging = True
                    cv2.putText(img, "DRAG START", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    speak("Drag start")
                else:
                    cv2.putText(img, "DRAGGING", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            else:
                if dragging:
                    pyautogui.mouseUp()
                    dragging = False
                    cv2.putText(img, "DRAG END", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    speak("Drag end")

            # Scroll gesture (works alongside other gestures)
            if is_scroll_gesture(hand_landmarks):
                scroll_amount = 0
                if index.y < middle.y:  # Scroll Down
                    scroll_amount = -40
                    cv2.putText(img, "SCROLL DOWN", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                    speak("Scrolling down")
                else:  # Scroll Up
                    scroll_amount = 40
                    cv2.putText(img, "SCROLL UP", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                    speak("Scrolling up")
                pyautogui.scroll(scroll_amount)

            # Zoom gesture
            if is_zoom_gesture(hand_landmarks):
                thumb_index_dist = ((thumb.x - index.x) ** 2 + (thumb.y - index.y) ** 2) ** 0.5
                if thumb_index_dist > 0.2:  # Spread fingers
                    cv2.putText(img, "ZOOM IN", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 0, 200), 2)
                    pyautogui.keyDown('ctrl')
                    pyautogui.scroll(100)  # Zoom in
                    pyautogui.keyUp('ctrl')
                    speak("Zoom in")
                elif thumb_index_dist < 0.1:  # Pinched fingers
                    cv2.putText(img, "ZOOM OUT", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 0, 200), 2)
                    pyautogui.keyDown('ctrl')
                    pyautogui.scroll(-100)  # Zoom out
                    pyautogui.keyUp('ctrl')
                    speak("Zoom out")

    # Display FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    cv2.putText(img, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Update last gesture
    if current_gesture and (time.time() - last_voice_time) > gesture_cooldown:
        last_gesture = current_gesture

    cv2.imshow("Gesture Mouse Controller", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
engine.stop()