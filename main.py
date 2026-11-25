import cv2
import mediapipe as mp

camera = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

while True:
    success, frame = camera.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    total_fingers = 0

    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            landmarks = hand_landmarks.landmark
            fingers = 0

            # Four fingers
            for tip in [8, 12, 16, 20]:
                if landmarks[tip].y < landmarks[tip - 2].y:
                    fingers += 1

            # Thumb
            hand_label = results.multi_handedness[i].classification[0].label

            if hand_label == "Right":
                if landmarks[4].x < landmarks[3].x:
                    fingers += 1
            else:
                if landmarks[4].x > landmarks[3].x:
                    fingers += 1

            total_fingers += fingers

    cv2.putText(
        frame,
        f"Total Fingers: {total_fingers}",
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Hand Gesture Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
hands.close()
cv2.destroyAllWindows()