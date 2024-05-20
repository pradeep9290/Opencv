import cv2
import mediapipe as mp
import hand_Detection  # Import your hand detection module
import math

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Set up MediaPipe drawing utilities
mp_drawing = mp.solutions.drawing_utils

# Function to calculate the angle of rotation of the hand (wrist)
def calculate_angle(hand_landmarks):
    wrist = hand_landmarks[mp_hands.HandLandmark.WRIST]
    thumb = hand_landmarks[mp_hands.HandLandmark.THUMB_TIP]
    middle_finger = hand_landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

    vector1 = (thumb.x - wrist.x, thumb.y - wrist.y)
    vector2 = (middle_finger.x - wrist.x, middle_finger.y - wrist.y)

    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
    magnitude1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
    magnitude2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)

    cosine_angle = dot_product / (magnitude1 * magnitude2)
    angle = math.acos(cosine_angle)
    angle = math.degrees(angle)

    return angle

# Function to perform hand detection
def detect_hands(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hand_Detection.detect_hands(rgb_frame)
    return results

# Capture video from the camera
cap = cv2.VideoCapture(0)

# Define gesture variables
previous_gesture = None

while True:
    ret, frame = cap.read()

    results = detect_hands(frame)

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            hand_landmarks = landmarks.landmark
            angle = calculate_angle(hand_landmarks)

            gesture = hand_Detection.recognize_gesture(angle)

            cv2.putText(frame, f"Gesture: {gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"Angle: {angle:.2f} degrees", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"Recognized Command: {gesture}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if gesture == "Stop" and previous_gesture != "Stop":
                print("Stop")
                # Send a stop command to the robot here (implement this part)
            elif gesture == "Move Forward" and previous_gesture != "Move Forward":
                print("Move Forward")
                # Send a move forward command to the robot here (implement this part)
            elif gesture == "Turn Right" and previous_gesture != "Turn Right":
                print("Turn Right")
                # Send a turn right command to the robot here (implement this part)
            elif gesture == "Turn Left" and previous_gesture != "Turn Left":
                print("Turn Left")
                # Send a turn left command to the robot here (implement this part)

            previous_gesture = gesture

    cv2.imshow('Hand Tracking', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
