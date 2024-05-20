import cv2
import mediapipe as mp

# Enable camera
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Set up MediaPipe drawing utilities
mp_drawing = mp.solutions.drawing_utils

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to RGB (MediaPipe requires RGB input)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hand
    results = hands.process(rgb_frame)

    # If hands are detected, draw landmarks on the frame
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the frame with hand tracking
    cv2.imshow('Hand Tracking', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
