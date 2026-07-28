from flask import Flask, render_template, Response, jsonify, request
import cv2
import mediapipe as mp
import numpy as np
import threading
import time
import json
import os
from dotenv import load_dotenv
from groq import Groq
load_dotenv()
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

app = Flask(__name__)
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)
counter = 0
stage = "UP"
posture = "GOOD"
exercise = "squat"
history = []
calories = 0.0
user_weight = 70
min_rep_angle = 180
last_rep_time = None
angle_buffer = []
plank_start_time = None
session_active = True

cap = None
cap_lock = threading.Lock()

# ================= CAMERA INIT =================
def init_camera():
    global cap

    if cap is not None:
        try:
            cap.release()
        except:
            pass

    backends = [
        (1, cv2.CAP_DSHOW),
        (1, cv2.CAP_MSMF),
        (1, 0),
    ]

    for idx, backend in backends:
        try:
            c = cv2.VideoCapture(idx, backend)

            if c.isOpened():
                c.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                c.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                c.set(cv2.CAP_PROP_FPS, 30)

                for _ in range(5):
                    c.read()

                cap = c
                print(f"Camera opened with backend={backend}")
                return

        except Exception as e:
            print(f"Camera backend {backend} failed: {e}")

    cap = None
    print("No camera found")

init_camera()

# ================= MODEL =================
base_options = python.BaseOptions(model_asset_path='pose_landmarker_heavy.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO
)
detector = vision.PoseLandmarker.create_from_options(options)

# ================= STORAGE =================
def save_history():
    with open("data.json", "w") as f:
        json.dump({
            "history": history
        }, f)

def load_history():
    global history
    if os.path.exists("data.json"):
        with open("data.json", "r") as f:
            data = json.load(f)
            history = data.get("history", [])

load_history()

# ================= HELPERS =================
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

def smooth_angle(angle):
    global angle_buffer
    angle_buffer.append(angle)
    if len(angle_buffer) > 5:
        angle_buffer.pop(0)
    return sum(angle_buffer) / len(angle_buffer)

def lm(landmarks, idx):
    return [landmarks[idx].x, landmarks[idx].y]

def session_summary(ex_name, count, cal, posture_state):
    if ex_name == "squat":
        return f"Squats * {count} | {round(cal,1)} kcal | {posture_state}"
    elif ex_name == "pushup":
        return f"Push-ups * {count} | {round(cal,1)} kcal | {posture_state}"
    elif ex_name == "jumpingjack":
        return f"Jumping Jacks * {count} | {round(cal,1)} kcal | {posture_state}"
    elif ex_name == "bicepcurl":
        return f"Bicep Curls * {count} | {round(cal,1)} kcal | {posture_state}"
    elif ex_name == "plank":
        return f"Plank * {count} sec | {round(cal,1)} kcal | {posture_state}"
    return f"{ex_name} * {count} | {round(cal,1)} kcal | {posture_state}"

VISIBILITY_MAP = {
    "squat": [11, 23, 25, 27],
    "pushup": [11, 13, 15],
    "jumpingjack": [11, 12, 13, 15, 16],
    "bicepcurl": [11, 13, 15],
    "plank": [11, 23, 25],
}

def visibility_ok(landmarks, exercise_name, threshold=0.35):
    indices = VISIBILITY_MAP.get(exercise_name, [11, 13, 15])
    return all(landmarks[i].visibility >= threshold for i in indices)

# ================= FRAME GENERATOR =================
def generate_frames():
    global counter, stage, posture, exercise, calories, plank_start_time, session_active, cap
    global min_rep_angle, user_weight
    while True:
        with cap_lock:
            if cap is None or not cap.isOpened():
                init_camera()
                time.sleep(0.3)
                continue
            success, frame = cap.read()

        if not success or frame is None:
            time.sleep(0.03)
            continue

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        timestamp_ms = int(time.time() * 1000)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        detected = False

        try:
            result = detector.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]

                if session_active and visibility_ok(landmarks, exercise):
                    detected = True

                    l_shoulder = lm(landmarks, 11)
                    l_hip = lm(landmarks, 23)
                    l_knee = lm(landmarks, 25)
                    l_ankle = lm(landmarks, 27)
                    l_elbow = lm(landmarks, 13)
                    l_wrist = lm(landmarks, 15)

                    r_shoulder = lm(landmarks, 12)
                    r_hip = lm(landmarks, 24)
                    r_knee = lm(landmarks, 26)
                    r_ankle = lm(landmarks, 28)
                    r_elbow = lm(landmarks, 14)
                    r_wrist = lm(landmarks, 16)

                    left_vis = landmarks[13].visibility
                    right_vis = landmarks[14].visibility

                    # ===== SQUAT =====
                    if exercise == "squat":
                        if landmarks[25].visibility >= landmarks[26].visibility:
                            knee_angle = smooth_angle(calculate_angle(l_hip, l_knee, l_ankle))
                            min_rep_angle = min(min_rep_angle, knee_angle)
                            back_angle = smooth_angle(calculate_angle(l_shoulder, l_hip, l_knee))
                            
                        else:
                            knee_angle = smooth_angle(calculate_angle(r_hip, r_knee, r_ankle))
                            min_rep_angle = min(min_rep_angle, knee_angle)
                            back_angle = smooth_angle(calculate_angle(r_shoulder, r_hip, r_knee))
                           

                        posture = "GOOD" if back_angle > 150 else "BAD"

                        if knee_angle < 105:
                            stage = "DOWN"
                        elif knee_angle > 158 and stage == "DOWN":
                            stage = "UP"
                            counter += 1
                            rom_factor = max(0.8, min(1.3, (160 - min_rep_angle) / 60))
                            weight_factor = user_weight / 70
                            calories += 0.5 * rom_factor * weight_factor
                            min_rep_angle = 180

                    # ===== PUSH-UP =====
                    elif exercise == "pushup":
                        if left_vis >= right_vis:
                            elbow_angle = smooth_angle(calculate_angle(l_shoulder, l_elbow, l_wrist))
                        else:
                            elbow_angle = smooth_angle(calculate_angle(r_shoulder, r_elbow, r_wrist))

                        posture = "GOOD"

                        if elbow_angle < 90:
                            stage = "DOWN"
                        elif elbow_angle > 150 and stage == "DOWN":
                            stage = "UP"
                            counter += 1
                            weight_factor = user_weight / 70
                            calories += 0.35 * weight_factor

                    # ===== JUMPING JACK =====
                    elif exercise == "jumpingjack":
                        l_hand_y = landmarks[15].y
                        r_hand_y = landmarks[16].y
                        shoulder_y = (landmarks[11].y + landmarks[12].y) / 2
                        leg_dist = abs(landmarks[27].x - landmarks[28].x)

                        hands_up = l_hand_y < (shoulder_y - 0.05) and r_hand_y < (shoulder_y - 0.05)
                        hands_down = l_hand_y > shoulder_y and r_hand_y > shoulder_y

                        posture = "GOOD"

                        if hands_up and leg_dist > 0.28:
                            stage = "OPEN"
                        elif hands_down and leg_dist < 0.22 and stage == "OPEN":
                            stage = "CLOSE"
                            counter += 1
                            weight_factor = user_weight / 70
                            calories += 0.45 * weight_factor

                    # ===== BICEP CURL =====
                    elif exercise == "bicepcurl":
                        if left_vis >= right_vis:
                            elbow_angle = smooth_angle(calculate_angle(l_shoulder, l_elbow, l_wrist))
                        else:
                            elbow_angle = smooth_angle(calculate_angle(r_shoulder, r_elbow, r_wrist))

                        posture = "GOOD"

                        if elbow_angle > 150:
                            stage = "DOWN"
                        elif elbow_angle < 55 and stage == "DOWN":
                            stage = "UP"
                            counter += 1
                            weight_factor = user_weight / 70
                            calories += 0.25 * weight_factor

                    # ===== PLANK =====
                    elif exercise == "plank":
                        if landmarks[23].visibility >= landmarks[24].visibility:
                            back_angle = smooth_angle(calculate_angle(l_shoulder, l_hip, l_knee))
                        else:
                            back_angle = smooth_angle(calculate_angle(r_shoulder, r_hip, r_knee))

                        if back_angle > 160:
                            posture = "GOOD"
                            if plank_start_time is None:
                                plank_start_time = time.time()
                                elapsed = time.time() - plank_start_time
                                counter = int(elapsed)
                                calories = (elapsed / 3600) * 3.5 * user_weight
                        else:
                            posture = "BAD"
                            plank_start_time = None

            if not detected:
                cv2.putText(
                    frame,
                    "Move closer / better lighting",
                    (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 200, 255),
                    2
                )

        except Exception as ex:
            print("Frame error:", ex)

        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# ================= ROUTES =================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_exercise/<name>')
def set_exercise(name):
    global exercise, counter, stage, posture, angle_buffer, plank_start_time, session_active, calories

    allowed = {"squat", "pushup", "jumpingjack", "bicepcurl", "plank"}
    if name not in allowed:
        return "Invalid", 400

    exercise = name
    counter = 0
    stage = "UP"
    posture = "GOOD"
    angle_buffer = []
    plank_start_time = None
    calories = 0.0
    session_active = True
    return "OK"

@app.route('/reset')
def reset():
    global counter, stage, posture, angle_buffer, calories, history, plank_start_time, session_active
    counter = 0
    stage = "UP"
    posture = "GOOD"
    angle_buffer = []
    calories = 0.0
    history = []
    plank_start_time = None
    session_active = True
    save_history()
    return "OK"

@app.route('/end_session')
def end_session():
    global counter, stage, posture, angle_buffer, plank_start_time, session_active, calories, history, exercise

    session_active = False

    if counter > 0:
        history.append(session_summary(exercise, counter, calories, posture))
        save_history()

    counter = 0
    stage = "UP"
    posture = "GOOD"
    angle_buffer = []
    plank_start_time = None
    calories = 0.0
    return "OK"

@app.route('/stats')
def stats():
    return jsonify({
        "reps": int(counter),
        "stage": stage,
        "posture": posture,
        "exercise": exercise,
        "calories": round(calories, 2)
    })

@app.route('/history')
def history_data():
    return jsonify({"history": history[-30:]})

@app.route("/chat", methods=["POST"])
def chat():

    data = request.json
    user_message = data.get("message", "")

    prompt = f"""
    You are FitCheck AI, a helpful fitness coach.

    Current workout:
    Exercise: {exercise}
    Reps: {counter}
    Posture: {posture}
    Calories: {round(calories, 2)}

    User Question:
    {user_message}

    Give concise and practical fitness advice.
    """

    try:

        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=500
        )

        reply = completion.choices[0].message.content

        return jsonify({
            "reply": reply
        })

    except Exception as e:

        return jsonify({
            "reply": f"Error: {str(e)}"
        })
if __name__ == "__main__":
    app.run(debug=False, threaded=True)