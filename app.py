import os
import time
import json
import logging
import threading
from collections import deque
import numpy as np
import cv2
from dotenv import load_dotenv
from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, session
from authlib.integrations.flask_client import OAuth
from werkzeug.middleware.proxy_fix import ProxyFix

# Import modular components
from exercise_tracker import WorkoutSessionState, evaluate_frame_pose, EXERCISE_LANDMARKS
from db_manager import DatabaseManager
from coach_service import run_fitness_agent
from agent_tools import FitnessAgentTools, validate_plan

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("fitcheck.app")

app = Flask(__name__)
# Ensure ProxyFix is applied so redirect_uri generates HTTPS behind Render reverse proxies
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1, x_prefix=1)

# Secret key validation
configured_secret = os.getenv("SECRET_KEY")
if not configured_secret:
    logger.warning("SECRET_KEY environment variable is not set. Generating temporary key for current runtime.")
app.secret_key = configured_secret or os.urandom(24).hex()

# Session cookie security settings
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.getenv("RENDER") is not None or os.getenv("FLASK_ENV") == "production"
)

# Initialize Database Manager
db_manager = DatabaseManager()

# OAuth Configuration
oauth = OAuth(app)
google_client_id = os.getenv("GOOGLE_CLIENT_ID")
google_client_secret = os.getenv("GOOGLE_CLIENT_SECRET")

if google_client_id and google_client_secret:
    google = oauth.register(
        name="google",
        client_id=google_client_id,
        client_secret=google_client_secret,
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={
            "scope": "openid email profile"
        }
    )
else:
    google = None
    logger.warning("Google OAuth credentials missing. Login route will show helpful guidance.")

# MediaPipe Pose Model
detector = None
detector_lock = threading.Lock()
try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    model_path = os.getenv("POSE_MODEL_PATH", "pose_landmarker_heavy.task")
    if os.path.exists(model_path):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO
        )
        detector = vision.PoseLandmarker.create_from_options(options)
        logger.info("MediaPipe PoseLandmarker loaded successfully.")
    else:
        logger.error("Pose model file not found at %s", model_path)
except Exception as e:
    logger.error("Error initializing MediaPipe PoseLandmarker: %s", e)

# Isolated User Session Store
# Maps user_key (email or session id) -> WorkoutSessionState
user_sessions = {}
sessions_lock = threading.Lock()
# Process-local, bounded coach memory. It is intentionally not stored in the
# signed Flask cookie and may be lost after a restart or differ across workers.
coach_conversations = {}
coach_conversations_lock = threading.Lock()

def get_current_user_key():
    """Identifies current session owner securely."""
    if "user" in session and session["user"].get("email"):
        return session["user"]["email"]
    if "anonymous_id" not in session:
        session["anonymous_id"] = os.urandom(12).hex()
    return session["anonymous_id"]

def get_user_state(user_key):
    """Retrieves or creates isolated workout state for a user."""
    with sessions_lock:
        if user_key not in user_sessions:
            initial_weight = 70.0
            if "user" in session and session["user"].get("email"):
                prof = db_manager.get_user_profile(session["user"]["email"])
                if prof and "weight" in prof:
                    initial_weight = float(prof["weight"])
            user_sessions[user_key] = WorkoutSessionState(exercise="squat", user_weight=initial_weight)
        return user_sessions[user_key]

# Tracking timestamp for video inference
last_detection_timestamp = 0
timestamp_lock = threading.Lock()

def format_session_summary(exercise, reps, calories, posture):
    if exercise == "plank":
        return f"Plank * {int(reps)} sec | {round(calories, 1)} kcal | {posture}"
    names = {
        "squat": "Squats",
        "pushup": "Push-ups",
        "jumpingjack": "Jumping Jacks",
        "bicepcurl": "Bicep Curls"
    }
    label = names.get(exercise, exercise.capitalize())
    return f"{label} * {int(reps)} | {round(calories, 1)} kcal | {posture}"

# ================= ROUTES =================

@app.route("/")
def index():
    if "user" not in session:
        return redirect(url_for("login"))
    user_key = get_current_user_key()
    user_state = get_user_state(user_key)
    return render_template(
        "index.html",
        user=session.get("user"),
        exercise=user_state.exercise,
        user_weight=user_state.user_weight
    )

@app.route("/login")
def login():
    if "user" in session:
        return redirect(url_for("index"))

    if not google:
        # Dev / offline mode fallback if OAuth is not configured
        return render_template(
            "login_prompt.html" if os.path.exists("templates/login_prompt.html") else "index.html",
            oauth_configured=False
        )

    # Note: _external=True generates full url using scheme configured by ProxyFix
    redirect_uri = url_for("auth_callback", _external=True)
    return google.authorize_redirect(redirect_uri)

@app.route("/dev_login", methods=["POST", "GET"])
def dev_login():
    """Allows testing when running in local development mode without Google credentials."""
    if os.getenv("FLASK_ENV") != "production" and not os.getenv("RENDER"):
        email = request.args.get("email", "dev_user@fitcheck.local")
        session["user"] = {
            "name": "Dev User",
            "email": email,
            "picture": ""
        }
        return redirect(url_for("index"))
    return "Not available in production", 403

@app.route("/login/callback")
def auth_callback():
    if not google:
        return "OAuth not configured", 500

    try:
        token = google.authorize_access_token()
        user_info = token.get("userinfo")
        if not user_info:
            user_info = google.userinfo()

        email = user_info.get("email")
        name = user_info.get("name", "User")
        session["user"] = {
            "name": name,
            "email": email,
            "picture": user_info.get("picture", "")
        }

        # Check existing profile in database
        prof = db_manager.get_user_profile(email)
        weight = float(prof["weight"]) if prof and "weight" in prof else 70.0
        db_manager.save_user_profile(email, name, weight)

        # Update or create session state with loaded profile weight
        user_state = get_user_state(email)
        user_state.user_weight = weight

        return redirect(url_for("index"))
    except Exception as e:
        logger.error("OAuth callback error: %s", e)
        return f"Authentication failed: {str(e)}", 400

@app.route("/logout")
def logout():
    user_key = get_current_user_key()
    with sessions_lock:
        if user_key in user_sessions:
            del user_sessions[user_key]
    with coach_conversations_lock:
        coach_conversations.pop(user_key, None)
    session.clear()
    return redirect(url_for("login"))

@app.route("/process_frame", methods=["POST"])
def process_frame():
    if "user" not in session:
        return jsonify({"error": "Unauthorized. Please log in."}), 401

    user_key = get_current_user_key()
    user_state = get_user_state(user_key)

    frame_bytes = request.get_data()
    if not frame_bytes:
        return jsonify({"error": "Empty frame data received"}), 400

    try:
        np_buffer = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "Invalid image payload"}), 400

        if detector is None:
            return jsonify({"error": "Pose detector model unavailable on server"}), 503

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        global last_detection_timestamp
        with timestamp_lock:
            timestamp_ms = int(time.time() * 1000)
            if timestamp_ms <= last_detection_timestamp:
                timestamp_ms = last_detection_timestamp + 1
            last_detection_timestamp = timestamp_ms

        with detector_lock:
            detection_result = detector.detect_for_video(mp_image, timestamp_ms)

        if not detection_result.pose_landmarks:
            # Body not detected in frame
            config = EXERCISE_LANDMARKS.get(user_state.exercise, EXERCISE_LANDMARKS["squat"])
            user_state.tracking_status = {"detected": False, "visibility_ok": False, "missing_landmarks": config["required"]}
            return jsonify({
                "detected": False,
                "reps": int(user_state.counter),
                "target_reps": user_state.target_reps,
                "stage": user_state.stage,
                "posture": "NOT_EVALUATED",
                "exercise": user_state.exercise,
                "calories": round(user_state.calories, 2),
                "guidance": "No person detected in frame. " + config["guidance"],
                "missing_landmarks": config["required"]
            }), 200

        # Person detected: run exercise evaluation
        landmarks = detection_result.pose_landmarks[0]
        diag = evaluate_frame_pose(user_state, landmarks, now=time.time())
        user_state.tracking_status = {
            "detected": True,
            "visibility_ok": diag.get("visibility_ok"),
            "missing_landmarks": diag.get("missing_landmarks", []),
        }

        resp = user_state.to_dict()
        resp["detected"] = True
        resp["visibility_ok"] = diag.get("visibility_ok", True)
        resp["primary_angle"] = diag.get("primary_angle")
        resp["guidance"] = diag.get("guidance")
        resp["missing_landmarks"] = diag.get("missing_landmarks", [])

        return jsonify(resp), 200

    except Exception as e:
        logger.error("Frame processing exception: %s", e)
        return jsonify({"error": f"Frame tracking error: {str(e)}"}), 500

@app.route("/set_exercise/<name>", methods=["POST", "GET"])
def set_exercise(name):
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    allowed = {"squat", "pushup", "jumpingjack", "bicepcurl", "plank"}
    if name not in allowed:
        return jsonify({"error": f"Invalid exercise. Choose from {list(allowed)}"}), 400

    user_key = get_current_user_key()
    user_state = get_user_state(user_key)
    user_state.reset_exercise_tracking(exercise=name)

    return jsonify({"ok": True, "exercise": user_state.exercise})

@app.route("/set_weight", methods=["POST"])
def set_weight():
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    try:
        data = request.get_json(silent=True) or {}
        val = float(data.get("weight", 70))
        if val < 20 or val > 200:
            return jsonify({"error": "Weight must be between 20 and 200 kg"}), 400

        user_key = get_current_user_key()
        user_state = get_user_state(user_key)
        user_state.user_weight = val

        # Save to database profile safely
        user_email = session["user"].get("email")
        user_name = session["user"].get("name", "User")
        db_manager.save_user_profile(user_email, user_name, val)

        return jsonify({"ok": True, "weight": user_state.user_weight})
    except (ValueError, TypeError):
        return jsonify({"error": "Please enter a valid numeric weight"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/reset", methods=["POST", "GET"])
def reset():
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    user_key = get_current_user_key()
    user_state = get_user_state(user_key)
    user_state.reset_exercise_tracking()

    # Clear user's workouts in MongoDB safely without deleting other users' data
    user_email = session["user"].get("email")
    db_manager.clear_user_workouts(user_email)

    return jsonify({"ok": True, "message": "Workout session and history reset"})

@app.route("/end_session", methods=["POST", "GET"])
def end_session():
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    user_key = get_current_user_key()
    user_state = get_user_state(user_key)
    user_state.session_active = False

    saved = False
    if user_state.counter > 0:
        summary_text = format_session_summary(
            user_state.exercise,
            user_state.counter,
            user_state.calories,
            user_state.posture
        )
        user_email = session["user"].get("email")
        session_data = {
            "exercise": user_state.exercise,
            "reps": int(user_state.counter),
            "calories": round(user_state.calories, 2),
            "posture": user_state.posture,
            "summary": summary_text,
            "timestamp": time.time()
        }
        saved = db_manager.save_completed_workout(user_email, session_data)

    user_state.reset_exercise_tracking()
    return jsonify({"ok": True, "saved": saved})

@app.route("/stats")
def stats():
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    user_key = get_current_user_key()
    user_state = get_user_state(user_key)
    return jsonify(user_state.to_dict())

@app.route("/history")
def history_data():
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    user_email = session["user"].get("email")
    workouts = db_manager.get_user_workouts(user_email, limit=30)
    summaries = [w.get("summary") for w in workouts if w.get("summary")]

    return jsonify({"history": summaries})

@app.route("/chat", methods=["POST"])
def chat():
    if "user" not in session:
        return jsonify({"reply": "Please log in to chat with the AI Fitness Coach."}), 401

    data = request.get_json(silent=True) or {}
    user_message = data.get("message", "")

    user_key = get_current_user_key()
    user_state = get_user_state(user_key)
    with coach_conversations_lock:
        conversation = list(coach_conversations.get(user_key, deque(maxlen=8)))
    tools = FitnessAgentTools(
        user_email=session["user"]["email"],
        user_state=user_state,
        db_manager=db_manager,
        tracking_status=getattr(user_state, "tracking_status", {}),
    )
    result = run_fitness_agent(user_message, tools, conversation=conversation)
    if result["status"] == 200:
        with coach_conversations_lock:
            memory = coach_conversations.setdefault(user_key, deque(maxlen=8))
            memory.append({"role": "user", "content": str(user_message)[:1200]})
            memory.append({"role": "assistant", "content": result["reply"][:1200]})
    return jsonify({
        "reply": result["reply"],
        "tool_activity": result["tool_activity"],
        "plan": result["plan"],
        "proposal": result["proposal"],
    }), result["status"]

@app.route("/plans", methods=["GET"])
def get_plans():
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    plans = db_manager.get_user_plans(session["user"]["email"], limit=10)
    if plans is None:
        return jsonify({"error": "Saved plans are unavailable because the database is not connected."}), 503
    return jsonify({"plans": plans})

@app.route("/plans", methods=["POST"])
def save_plan():
    """Persist only a plan explicitly submitted by the authenticated user."""
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    plan, error = validate_plan((request.get_json(silent=True) or {}).get("plan"))
    if error:
        return jsonify({"error": error}), 400
    if not db_manager.is_connected:
        return jsonify({"error": "Plan was not saved because the database is unavailable."}), 503
    if not db_manager.save_user_plan(session["user"]["email"], plan):
        return jsonify({"error": "Plan could not be saved."}), 503
    return jsonify({"ok": True, "plan": plan}), 201

@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "database_connected": db_manager.ping(),
        "pose_model_loaded": detector is not None
    })

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
