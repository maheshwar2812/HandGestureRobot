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


# Start MediaPipe
with HandLandmarker.create_from_options(options) as landmarker:

    camera = cv2.VideoCapture(0)

    while True:

        success, frame = camera.read()

        if not success:
            print("Camera not found")
            break

        # Flip camera so it behaves like a mirror
        frame = cv2.flip(frame, 1)

        # Convert BGR → RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Convert to MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Detect hand
        result = landmarker.detect(mp_image)

        if result.hand_landmarks:

            hand = result.hand_landmarks[0]

            # Get image dimensions
            height, width, _ = frame.shape

            # Draw landmarks
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

            # Finger landmark numbers
            finger_tips = [8, 12, 16, 20]

            fingers_up = 0

            # Check index, middle, ring and little finger
            for tip in finger_tips:

                if hand[tip].y < hand[tip - 2].y:
                    fingers_up += 1

            # Display number of fingers
            cv2.putText(
                frame,
                f"Fingers: {fingers_up}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        # Show camera
        cv2.imshow(
            "Finger Detection",
            frame
        )

        # Press Q to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()