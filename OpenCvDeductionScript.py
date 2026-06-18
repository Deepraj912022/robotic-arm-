import cv2
import numpy as np
from ultralytics import YOLO
from enum import Enum
import socket
import time
import json

# ================= UDP CONFIG =================
UDP_IP = "127.0.0.1"
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
last_sent_time = 0
SEND_INTERVAL = 20  # seconds

# ================= MEDIAPIPE =================
MEDIAPIPE_AVAILABLE = False
try:
    import mediapipe as mp
    if hasattr(mp, 'solutions'):
        MEDIAPIPE_AVAILABLE = True
        print("[INFO] MediaPipe loaded")
except:
    print("[INFO] MediaPipe not available")

# ================= CONFIG =================
CONF_THRESHOLD = 0.40
MODEL_NAME = "yolov8n-pose.pt"
HAND_MODEL_PATH = "yolov8n_hands.pt"

WRIST_INDICES = {'left': 9, 'right': 10}

class TrackingMode(Enum):
    YOLO_HANDS = "YOLO Hands"
    YOLO_POSE = "YOLO Pose (Wrist)"
    if MEDIAPIPE_AVAILABLE:
        MEDIAPIPE = "MediaPipe Full Hand"

# ================= LOAD HAND MODEL =================
def load_hand_model():
    try:
        return YOLO(HAND_MODEL_PATH)
    except:
        print("[WARN] Hand model not found")
        return None

# ================= YOLO HANDS =================
def process_yolo_hands(frame, model):
    coords = []
    results = model(frame, conf=CONF_THRESHOLD, verbose=False)[0]

    if results.boxes is not None:
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            coords.append({
                "type": "hand_box",
                "x": cx,
                "y": cy
            })

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,200,100), 2)
            cv2.circle(frame, (cx, cy), 5, (255,255,0), -1)

    return coords

# ================= YOLO POSE =================
def process_yolo_pose(frame, model):
    coords = []
    results = model(frame, conf=CONF_THRESHOLD, verbose=False)[0]

    if results.keypoints is not None:
        for kp in results.keypoints.data:
            for hand_name, idx in WRIST_INDICES.items():
                if idx < len(kp):
                    try:
                        x, y, conf = kp[idx].tolist()
                    except:
                        continue

                    if conf > 0.3:
                        x, y = int(x), int(y)

                        coords.append({
                            "type": "wrist",
                            "hand": hand_name,
                            "x": x,
                            "y": y
                        })

                        cv2.circle(frame, (x, y), 8, (255,100,0), -1)
                        cv2.putText(frame, f"{hand_name}",
                                    (x+10, y-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0,255,0), 1)

    return coords

# ================= MEDIAPIPE FULL HAND =================
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20)
]

def process_mediapipe(frame, hands, rgb):
    coords = []
    h, w, _ = frame.shape

    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            hand_points = []

            # Draw connections
            for c in HAND_CONNECTIONS:
                s = hand_landmarks.landmark[c[0]]
                e = hand_landmarks.landmark[c[1]]

                p1 = (int(s.x * w), int(s.y * h))
                p2 = (int(e.x * w), int(e.y * h))

                cv2.line(frame, p1, p2, (100,100,255), 2)

            # Draw landmarks + collect coords
            for i, lm in enumerate(hand_landmarks.landmark):
                x, y = int(lm.x * w), int(lm.y * h)

                hand_points.append({"id": i, "x": x, "y": y})

                if i == 0:
                    cv2.circle(frame, (x,y), 6, (255,100,0), -1)
                else:
                    cv2.circle(frame, (x,y), 4, (0,150,255), -1)

            coords.append({
                "type": "mediapipe_hand",
                "landmarks": hand_points
            })

    return coords

# ================= MAIN =================
def main():
    global last_sent_time

    print("[INFO] Loading models...")
    hand_model = load_hand_model()
    pose_model = YOLO(MODEL_NAME)

    hands = None
    if MEDIAPIPE_AVAILABLE:
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(max_num_hands=2)

    cap = cv2.VideoCapture(0)

    modes = [TrackingMode.YOLO_HANDS, TrackingMode.YOLO_POSE]
    if hands:
        modes.append(TrackingMode.MEDIAPIPE)

    mode_index = 1
    mode = modes[mode_index]

    print("[INFO] Press T to switch modes")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        all_coords = []

        # ================= MODES =================
        if mode == TrackingMode.YOLO_HANDS and hand_model:
            all_coords = process_yolo_hands(frame, hand_model)

        elif mode == TrackingMode.YOLO_POSE:
            all_coords = process_yolo_pose(frame, pose_model)

        elif mode == TrackingMode.MEDIAPIPE and hands:
            all_coords = process_mediapipe(frame, hands, rgb)

        # ================= UDP SEND =================
        current_time = time.time()

        if current_time - last_sent_time >= SEND_INTERVAL and all_coords:
            payload = {
                "mode": mode.value,
                "timestamp": current_time,
                "coords": all_coords
            }

            sock.sendto(json.dumps(payload).encode(), (UDP_IP, UDP_PORT))
            print("[UDP SENT]", payload)

            last_sent_time = current_time

        # ================= DISPLAY =================
        cv2.putText(frame, f"Mode: {mode.value}",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0,255,255), 2)

        cv2.imshow("Tracker", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('t'):
            mode_index = (mode_index + 1) % len(modes)
            mode = modes[mode_index]
            print("[INFO] Mode:", mode.value)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
