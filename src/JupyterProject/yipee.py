import cv2
import mediapipe as mp
import time
from pylsl import StreamInfo, StreamOutlet

# --- 1. Setup LSL ---
# (Stream Name, Type, Channel Count, Nominal Srate, Format, Unique ID)
info = StreamInfo('GestureMarkers', 'Markers', 1, 0, 'string', 'my_gesture_id_123')
outlet = StreamOutlet(info)

# --- 2. Setup MediaPipe ---
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='gesture_recognizer.task'),
    running_mode=VisionRunningMode.VIDEO
)

with GestureRecognizer.create_from_options(options) as recognizer:
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        timestamp = int(time.time() * 1000)
        result = recognizer.recognize_for_video(mp_image, timestamp)

        if result.gestures:
            top_gesture = result.gestures[0][0].category_name
            # Push the gesture name to the LSL network
            outlet.push_sample([top_gesture])
            print(f"Broadcasting: {top_gesture}")

        cv2.imshow('Sender', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()