# 🏋️ FitCheck AI

FitCheck AI is an AI-powered fitness assistant that uses real-time computer vision to track exercises, count repetitions, analyze posture, estimate calories burned, and provide fitness guidance.

The application uses **MediaPipe Pose Landmarker** to detect body landmarks through the user's webcam and analyzes joint angles to identify different stages of an exercise.

## ✨ Features

- 📷 Real-time webcam-based pose detection
- 🔢 Automatic repetition counting
- 🧍 Posture analysis and feedback
- 🔥 Estimated calorie tracking based on user weight
- 📊 Workout session history
- 🤖 AI-powered fitness coach
- 🏃 Support for multiple exercises

### Supported Exercises

- Squats
- Push-ups
- Jumping Jacks
- Bicep Curls
- Plank

## 🧠 How It Works

FitCheck AI captures video from the user's webcam and processes each frame using **MediaPipe Pose Landmarker**.

Body landmarks such as the shoulders, elbows, hips, knees, ankles, and wrists are detected.

The application calculates joint angles from these landmarks and uses exercise-specific rules to determine the current movement stage.

For example, during a squat:

1. The hip, knee, and ankle landmarks are detected.
2. The knee angle is calculated.
3. A sufficiently low knee angle identifies the downward phase.
4. Returning to the standing position completes the repetition.
5. The repetition counter is increased.

Similar movement logic is used for the other supported exercises.

## 🛠️ Technologies Used

### Backend

- Python
- Flask

### Computer Vision

- OpenCV
- MediaPipe
- NumPy

### Frontend

- HTML
- CSS
- JavaScript

### AI Integration

- Groq API
- Llama 3.3 70B

## 📁 Project Structure

```text
FitCheck-AI/
│
├── app.py
├── data.json
├── pose_landmarker_heavy.task
├── requirements.txt
├── .gitignore
│
└── templates/
    └── index.html
```

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/tarang54-tech/FitCheck-AI.git
```

Move into the project directory:

```bash
cd FitCheck-AI
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the Groq API key

Create a `.env` file inside the project directory:

```text
GROQ_API_KEY=your_groq_api_key
```

The `.env` file is excluded from Git using `.gitignore`, so API credentials are not stored in the repository.

### 5. Run FitCheck AI

```bash
python app.py
```

Open the local Flask address shown in the terminal, typically:

```text
http://127.0.0.1:5000
```

Allow camera access when required.

## 🤖 AI Fitness Coach

FitCheck AI includes an AI fitness assistant that can provide concise workout guidance based on the current exercise session.

The assistant can access information such as:

- Current exercise
- Repetition count
- Current posture status
- Estimated calories burned

This allows the chatbot to provide advice relevant to the ongoing workout.

## 🔥 Calorie Estimation

FitCheck AI provides an estimated calorie count based on factors including:

- Exercise performed
- Number of repetitions or exercise duration
- User body weight
- Exercise movement characteristics

The calorie values are estimates and should not be considered medical measurements.

## 🎯 Project Objective

The goal of FitCheck AI is to explore how computer vision and AI can be combined to create an interactive fitness assistant that provides real-time exercise tracking without requiring dedicated wearable hardware.

## 🔮 Future Improvements

Possible future additions include:

- More exercises
- Improved posture classification
- User authentication
- Personal workout profiles
- Workout analytics and progress graphs
- Database-based workout history
- More advanced calorie estimation
- Personalized AI workout plans
- Improved exercise-form feedback

## ⚠️ Disclaimer

FitCheck AI is an educational project and is not intended to provide medical advice or replace professional fitness guidance.

## 👨‍💻 Author

Developed by **Tarang**

Computer Science Engineering Student  
VIT Chennai
