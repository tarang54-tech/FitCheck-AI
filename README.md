# 🏋️ FitCheck AI

FitCheck AI is an intelligent computer-vision fitness assistant that uses real-time pose tracking to detect exercise movements, count repetitions accurately, analyze posture, estimate calories burned, persist workout history, and provide an AI fitness coach powered by Groq.

The application captures video frames from the user's browser, applies **MediaPipe Pose Landmarker**, evaluates joint angles using independent angle smoothing and robust state machines, and provides real-time body framing guidance.

---

## ✨ Features

- 📷 **Real-time browser webcam pose tracking** with MediaPipe Pose Landmarker Heavy.
- 🔢 **Deterministic repetition counting** with dedicated state machines for Squats, Push-ups, Jumping Jacks, and Bicep Curls.
- 🧘 **Plank timer & posture tracking** that counts elapsed hold time only while proper body alignment is maintained.
- 📐 **Independent angle smoothing** preventing joint cross-talk between knees, elbows, and back angles.
- 🧍 **Real-time body framing guidance** indicating when key joints are out of frame or obscured.
- 🔒 **User isolation & atomic MongoDB persistence** separating user profiles from workout session history.
- 🤖 **AI Fitness Coach** powered by Groq (Llama 3.3 70B & fast fallbacks) with context-aware exercise guidance and robust error handling.
- 🔐 **Google OAuth 2.0 authentication** with CSRF state protection and HTTPS reverse proxy compatibility.

---

## 🏃 Supported Exercises & Movement Cycles

| Exercise | Primary Joints Tracked | Movement State Cycle | Good Posture Criteria |
| :--- | :--- | :--- | :--- |
| **Squats** | Hip, Knee, Ankle | `UP` -> `DOWN` (knee < 110°) -> `UP` (knee > 155°) | Torso/back angle ≥ 145° (chest up) |
| **Push-ups** | Shoulder, Elbow, Wrist | `UP` -> `DOWN` (elbow < 90°) -> `UP` (elbow > 150°) | Body line (shoulder-hip-ankle) ≥ 150° |
| **Bicep Curls** | Shoulder, Elbow, Wrist | `DOWN` -> `UP` (elbow < 60°) -> `DOWN` (elbow > 145°) | Controlled upper arm movement |
| **Jumping Jacks** | Wrist, Shoulder, Ankle | `CLOSE` -> `OPEN` (hands up + feet wide) -> `CLOSE` | Synchronized full range of motion |
| **Plank** | Shoulder, Hip, Ankle | Active hold timer (seconds) | Straight body alignment (158° – 180°) |

---

## 🛠️ Technology Stack

- **Backend:** Python 3.11, Flask, Gunicorn
- **Computer Vision:** OpenCV, MediaPipe Pose Landmarker, NumPy
- **Frontend:** Vanilla HTML5, CSS3, modern asynchronous JavaScript
- **Database:** MongoDB Atlas (PyMongo) with isolated profile and workout collections
- **Authentication:** Google OAuth via Authlib & Flask Session
- **AI Coach:** Groq Cloud API (`llama-3.3-70b-versatile`)
- **Deployment:** Render (Docker / Linux container)

---

## 📁 Repository Structure

```text
FitCheck-AI/
├── app.py                     # Main Flask web application, route handlers & session tracking
├── exercise_tracker.py        # Independent angle smoothing, visibility checks & exercise state machines
├── db_manager.py              # MongoDB Atlas connection manager & atomic profile/history isolation
├── coach_service.py           # Groq API client with context injection and fallback models
├── pose_landmarker_heavy.task # MediaPipe Pose Landmarker binary model
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Render container configuration
├── .env.example               # Environment variables template
├── templates/
│   └── index.html             # Dashboard UI with live video, framing hints & AI chat
└── tests/
    ├── test_exercise_tracker.py
    ├── test_coach_service.py
    ├── test_db_manager.py
    └── test_routes.py
```

---

## 🚀 Local Setup & Development

### 1. Prerequisites
- Python 3.11 installed
- Git installed
- A webcam connected to your computer

### 2. Clone the repository
```bash
git clone https://github.com/tarang54-tech/FitCheck-AI.git
cd FitCheck-AI
```

### 3. Create a virtual environment
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Configure environment variables
Copy `.env.example` to `.env` and fill in your credentials:
```bash
cp .env.example .env
```

Set the following variables inside `.env`:
- `SECRET_KEY`: A random secret key for Flask sessions.
- `MONGO_URI`: Your MongoDB Atlas cluster connection string.
- `GROQ_API_KEY`: Your API key from [Groq Console](https://console.groq.com/keys).
- `GOOGLE_CLIENT_ID`: Your Google OAuth 2.0 Web Client ID.
- `GOOGLE_CLIENT_SECRET`: Your Google OAuth 2.0 Web Client Secret.

> **Local Google OAuth Redirect URI:** In Google Cloud Console, add `http://localhost:5000/login/callback` (or `http://127.0.0.1:5000/login/callback`) to your Authorized Redirect URIs.

### 6. Run automated tests
```bash
python -m unittest discover -s tests
```

### 7. Start the application
```bash
python app.py
```
Open your browser and navigate to `http://localhost:5000`.

---

## ☁️ Deployment on Render

### Required Environment Variables in Render Dashboard
Go to your Render Web Service -> **Environment** and add:

| Variable | Description | Provider |
| :--- | :--- | :--- |
| `SECRET_KEY` | Persistent random secret string (e.g. 32-hex chars) | Flask / Application |
| `MONGO_URI` | MongoDB Atlas SRV URI with read/write access | MongoDB Atlas |
| `GROQ_API_KEY` | Valid API key starting with `gsk_` | Groq Cloud |
| `GOOGLE_CLIENT_ID` | OAuth 2.0 Web Application Client ID | Google Cloud Console |
| `GOOGLE_CLIENT_SECRET` | OAuth 2.0 Web Application Client Secret | Google Cloud Console |
| `PORT` | Set automatically by Render (usually `10000`) | Render |

### Google OAuth Redirect URI for Render
In Google Cloud Console under your OAuth 2.0 Client ID settings, add your Render domain callback to **Authorized redirect URIs**:
```text
https://<your-service-name>.onrender.com/login/callback
```
*(e.g., `https://fitcheck-ai-l34e.onrender.com/login/callback`)*

---

## 🧪 Testing Checklist

- [x] **Squats:** Verify knee angle calculation, rep count increment on complete cycle (UP -> DOWN -> UP), and chest form check.
- [x] **Push-ups:** Verify elbow angle calculation, rep count increment on complete cycle (UP -> DOWN -> UP), and straight body line check.
- [x] **Bicep Curls:** Verify start in DOWN, curl to UP, and completion upon return to DOWN.
- [x] **Jumping Jacks:** Verify CLOSE -> OPEN -> CLOSE movement cycle.
- [x] **Plank:** Verify active second timer when body alignment is good; verify pause/hold when posture sags or peaks.
- [x] **Framing Guidance:** Verify clear message when stepping partially out of frame.
- [x] **AI Fitness Coach:** Verify response formatting, context injection, and graceful handling of missing or invalid API keys.
- [x] **Database Isolation:** Verify user workouts save to user's history and do not delete unrelated user collections or profiles.
- [x] **Google OAuth:** Verify login redirect, token exchange, and persistent profile weight retrieval.
