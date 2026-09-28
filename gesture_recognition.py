import cv2
import mediapipe as mp

# MediaPipe setup
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="models/hand_landmarker.task"
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)

# Start camera
camera = cv2.VideoCapture(0)

with HandLandmarker.create_from_options(options) as landmarker:

    while True:

        success, frame = camera.read()

        if not success:
            print("Camera not found")
            break

        # Convert BGR → RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Detect hand
        result = landmarker.detect(mp_image)

        gesture = "NO HAND"

        if result.hand_landmarks:

            hand = result.hand_landmarks[0]

            # Count fingers
            fingers_up = 0

            # Index finger
            if hand[8].y < hand[6].y:
                fingers_up += 1

            # Middle finger
            if hand[12].y < hand[10].y:
                fingers_up += 1

            # Ring finger
            if hand[16].y < hand[14].y:
                fingers_up += 1

            # Little finger
            if hand[20].y < hand[18].y:
                fingers_up += 1

            # Gesture mapping
            if fingers_up == 0:
                gesture = "STOP"

            elif fingers_up == 1:
                gesture = "FORWARD"

            elif fingers_up == 2:
                gesture = "BACKWARD"

            elif fingers_up == 3:
                gesture = "LEFT"

            elif fingers_up == 4:
                gesture = "RIGHT"

            # Draw landmarks
            height, width, _ = frame.shape

            for landmark in hand:
                x = int(landmark.x * width)
                y = int(landmark.y * height)

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )

        # Display gesture
        cv2.putText(
            frame,
            "COMMAND: " + gesture,
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3
        )

        cv2.imshow(
            "Hand Gesture Robot Control",
            frame
        )

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()