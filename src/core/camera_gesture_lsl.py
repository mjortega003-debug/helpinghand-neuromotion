import cv2
import mediapipe as mp
import time
import json
from pylsl import StreamInfo, StreamOutlet

# LSL SETUP
info = StreamInfo('CV_Stream', 'Markers', 1, 0, 'string', 'cam_gesture_001')
outlet = StreamOutlet(info)

# MEDIAPIPE SETUP 
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=r'src\core\gesture_recognizer.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1 # Force it to only track one hand for simplicity
)

# Thresholds
CONFIDENCE_THRESHOLD = 0.80
TARGET_HAND = "Right" 

def run_camera_lsl():
    cap = cv2.VideoCapture(0)
    
    with GestureRecognizer.create_from_options(options) as recognizer:
        print("[Camera] Rich Data LSL Outlet started. Broadcasting...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            frame_timestamp_ms = int(time.time() * 1000)
            
            result = recognizer.recognize_for_video(mp_image, frame_timestamp_ms)

            # DATA EXTRACTION & FILTERING
            payload = {
                "label": "none",
                "score": 0.0,
                "handedness": "unknown",
                "landmarks": [] # Will hold 63 floats (21 points * 3 axes)
            }


            if result.gestures and result.hand_landmarks and result.handedness:
                gesture = result.gestures[0][0]
                handedness = result.handedness[0][0].category_name
                landmarks = result.hand_landmarks[0]
                

                if handedness == TARGET_HAND and gesture.score >= CONFIDENCE_THRESHOLD:
                    payload["label"] = gesture.category_name
                    payload["score"] = round(gesture.score, 4)
                    payload["handedness"] = handedness
                    
                    # Extract the 21 3D coordinates (x, y, z)
                    for lm in landmarks:
                        payload["landmarks"].extend([round(lm.x, 4), round(lm.y, 4), round(lm.z, 4)])


            json_payload = json.dumps(payload)
            outlet.push_sample([json_payload])

            #VISUAL FEEDBACK (we can remove this later, but it's helpful for debugging the CV stream))
            color = (0, 255, 0) if payload["label"] != "none" else (0, 0, 255)
            display_text = f"{payload['handedness']} | {payload['label']} ({payload['score']:.2f})"
            
            cv2.putText(frame, display_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.imshow('HelpingHand CV Feed', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_camera_lsl()