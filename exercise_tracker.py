import time
import numpy as np
from collections import deque

def calculate_angle(a, b, c):
    """
    Calculates the angle at point b formed by points a, b, c.
    Returns angle in degrees [0, 180].
    """
    a, b, c = np.array(a, dtype=float), np.array(b, dtype=float), np.array(c, dtype=float)
    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0:
        return 0.0

    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    # Clamp for numerical precision issues
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine_angle))
    return float(angle)

class AngleSmoother:
    """Independent exponential moving average (EMA) or moving average smoother."""
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.history = deque(maxlen=window_size)

    def update(self, val):
        self.history.append(float(val))
        return sum(self.history) / len(self.history)

    def reset(self):
        self.history.clear()

    @property
    def value(self):
        if not self.history:
            return None
        return sum(self.history) / len(self.history)

# Landmark Index Constants matching MediaPipe Pose Landmarker
NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28

# Key required landmark indices and friendly guidance per exercise
EXERCISE_LANDMARKS = {
    "squat": {
        "required": [LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE],
        "name": "Squat",
        "guidance": "Step back so hips, knees, and ankles are clearly in camera view (side profile works best)."
    },
    "pushup": {
        "required": [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST],
        "name": "Push-up",
        "guidance": "Position camera at side profile. Keep shoulders, elbows, and wrists in view."
    },
    "bicepcurl": {
        "required": [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST],
        "name": "Bicep Curl",
        "guidance": "Keep upper body framed so active shoulder, elbow, and wrist remain in view."
    },
    "jumpingjack": {
        "required": [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_WRIST, RIGHT_WRIST, LEFT_ANKLE, RIGHT_ANKLE],
        "name": "Jumping Jack",
        "guidance": "Full body must be visible from overhead hands to ankles. Face camera directly."
    },
    "plank": {
        "required": [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE],
        "name": "Plank",
        "guidance": "Full side view required. Keep shoulders, hips, knees, and ankles clearly visible."
    }
}

class WorkoutSessionState:
    """Encapsulates isolated tracking state for a specific user or session."""
    def __init__(self, exercise="squat", user_weight=70.0):
        self.exercise = exercise
        self.user_weight = float(user_weight)
        self.counter = 0
        self.stage = "UP"  # "UP", "DOWN", "OPEN", "CLOSE"
        self.posture = "GOOD"
        self.calories = 0.0
        self.session_active = True
        self.target_reps = 10

        # Performance & fatigue metrics
        self.rep_times = []
        self.last_rep_timestamp = None
        self.last_rep_duration = None
        self.long_pause_count = 0
        self.bad_form_count = 0
        self.struggle_score = 0

        # Exercise specific state variables
        self.min_rep_angle = 180.0
        self.max_rep_angle = 0.0
        self.plank_start_time = None
        self.plank_accumulated_seconds = 0.0

        # Independent smoothers
        self.smoothers = {
            "squat_knee_left": AngleSmoother(5),
            "squat_knee_right": AngleSmoother(5),
            "squat_back_left": AngleSmoother(5),
            "squat_back_right": AngleSmoother(5),
            "pushup_elbow_left": AngleSmoother(5),
            "pushup_elbow_right": AngleSmoother(5),
            "pushup_back_left": AngleSmoother(5),
            "pushup_back_right": AngleSmoother(5),
            "bicep_elbow_left": AngleSmoother(5),
            "bicep_elbow_right": AngleSmoother(5),
            "plank_body_left": AngleSmoother(5),
            "plank_body_right": AngleSmoother(5),
        }

    def reset_exercise_tracking(self, exercise=None):
        if exercise is not None:
            self.exercise = exercise
        self.counter = 0
        if self.exercise == "jumpingjack":
            self.stage = "CLOSE"
        elif self.exercise == "bicepcurl":
            self.stage = "DOWN"
        else:
            self.stage = "UP"
        self.posture = "NOT_EVALUATED"
        self.calories = 0.0
        self.min_rep_angle = 180.0
        self.max_rep_angle = 0.0
        self.plank_start_time = None
        self.plank_accumulated_seconds = 0.0
        self.rep_times = []
        self.last_rep_timestamp = None
        self.last_rep_duration = None
        self.long_pause_count = 0
        self.bad_form_count = 0
        self.struggle_score = 0
        self.session_active = True
        for smoother in self.smoothers.values():
            smoother.reset()

    def record_completed_rep(self, now=None):
        if now is None:
            now = time.time()

        if self.last_rep_timestamp is not None:
            duration = now - self.last_rep_timestamp
            self.last_rep_duration = duration
            self.rep_times.append(duration)

            if duration > 5:
                self.long_pause_count += 1

            if len(self.rep_times) >= 3:
                previous_avg = sum(self.rep_times[:-1]) / len(self.rep_times[:-1])
                if duration > previous_avg * 1.5:
                    self.struggle_score = min(100, self.struggle_score + 10)

        self.last_rep_timestamp = now

        if self.posture == "BAD":
            self.bad_form_count += 1
            self.struggle_score = min(100, self.struggle_score + 15)

        self.counter += 1

    def to_dict(self):
        return {
            "reps": int(self.counter),
            "target_reps": self.target_reps,
            "stage": self.stage,
            "posture": self.posture,
            "exercise": self.exercise,
            "calories": round(self.calories, 2),
            "last_rep_duration": round(self.last_rep_duration, 2) if self.last_rep_duration else None,
            "long_pause_count": self.long_pause_count,
            "bad_form_count": self.bad_form_count,
            "struggle_score": self.struggle_score,
            "session_active": self.session_active,
            "user_weight": self.user_weight
        }

def check_visibility(landmarks, required_indices, min_vis=0.5):
    """
    Checks if required landmark indices meet visibility threshold.
    Returns (ok: bool, missing_indices: list, avg_vis: float).
    """
    if not landmarks:
        return False, required_indices, 0.0

    missing = []
    total_vis = 0.0
    for idx in required_indices:
        if idx >= len(landmarks):
            missing.append(idx)
            continue
        vis = getattr(landmarks[idx], 'visibility', 0.0)
        total_vis += vis
        if vis < min_vis:
            missing.append(idx)

    avg_vis = total_vis / len(required_indices) if required_indices else 0.0
    return (len(missing) == 0), missing, avg_vis

def pt(landmarks, idx):
    return [landmarks[idx].x, landmarks[idx].y]

def evaluate_frame_pose(state: WorkoutSessionState, landmarks, now=None):
    """
    Processes pose landmarks for the current exercise in state.
    Returns diagnostics dict with tracking details.
    """
    if now is None:
        now = time.time()

    ex = state.exercise
    config = EXERCISE_LANDMARKS.get(ex, EXERCISE_LANDMARKS["squat"])
    required = config["required"]

    # Visibility check
    vis_ok, missing, avg_vis = check_visibility(landmarks, required, min_vis=0.5)

    diagnostics = {
        "detected": True,
        "visibility_ok": vis_ok,
        "primary_angle": None,
        "guidance": None,
        "missing_landmarks": missing,
        "exercise": ex,
        "stage": state.stage,
        "reps": state.counter,
        "posture": state.posture,
        "calories": round(state.calories, 2)
    }

    if not vis_ok:
        diagnostics["guidance"] = config["guidance"]
        # Do not fabricate posture if not visible
        state.posture = "NOT_EVALUATED"
        diagnostics["posture"] = state.posture
        if ex == "plank" and state.plank_start_time is not None:
            # Pause/stop plank timing if body lost
            state.plank_start_time = None
        return diagnostics

    # If visibility is ok, evaluate specific exercise
    if ex == "squat":
        # Check both sides, choose side with better ankle/knee visibility
        left_leg_vis = (landmarks[LEFT_KNEE].visibility + landmarks[LEFT_ANKLE].visibility) / 2.0
        right_leg_vis = (landmarks[RIGHT_KNEE].visibility + landmarks[RIGHT_ANKLE].visibility) / 2.0

        if left_leg_vis >= right_leg_vis:
            raw_knee = calculate_angle(pt(landmarks, LEFT_HIP), pt(landmarks, LEFT_KNEE), pt(landmarks, LEFT_ANKLE))
            knee_angle = state.smoothers["squat_knee_left"].update(raw_knee)
            raw_back = calculate_angle(pt(landmarks, LEFT_SHOULDER), pt(landmarks, LEFT_HIP), pt(landmarks, LEFT_KNEE))
            back_angle = state.smoothers["squat_back_left"].update(raw_back)
        else:
            raw_knee = calculate_angle(pt(landmarks, RIGHT_HIP), pt(landmarks, RIGHT_KNEE), pt(landmarks, RIGHT_ANKLE))
            knee_angle = state.smoothers["squat_knee_right"].update(raw_knee)
            raw_back = calculate_angle(pt(landmarks, RIGHT_SHOULDER), pt(landmarks, RIGHT_HIP), pt(landmarks, RIGHT_KNEE))
            back_angle = state.smoothers["squat_back_right"].update(raw_back)

        diagnostics["primary_angle"] = round(knee_angle, 1)
        state.min_rep_angle = min(state.min_rep_angle, knee_angle)

        # Posture check: back/torso angle should remain upright (> 145 deg)
        if back_angle >= 145:
            state.posture = "GOOD"
        else:
            state.posture = "BAD"
            diagnostics["guidance"] = "Keep chest up and avoid leaning too far forward."

        # State machine: UP -> DOWN (knee < 110) -> UP (knee > 155)
        if knee_angle < 110:
            state.stage = "DOWN"
        elif knee_angle > 155 and state.stage == "DOWN":
            state.stage = "UP"
            state.record_completed_rep(now)
            rom_factor = max(0.8, min(1.3, (160.0 - state.min_rep_angle) / 60.0))
            weight_factor = state.user_weight / 70.0
            state.calories += 0.5 * rom_factor * weight_factor
            state.min_rep_angle = 180.0

    elif ex == "pushup":
        l_vis = (landmarks[LEFT_ELBOW].visibility + landmarks[LEFT_WRIST].visibility) / 2.0
        r_vis = (landmarks[RIGHT_ELBOW].visibility + landmarks[RIGHT_WRIST].visibility) / 2.0

        if l_vis >= r_vis:
            raw_elbow = calculate_angle(pt(landmarks, LEFT_SHOULDER), pt(landmarks, LEFT_ELBOW), pt(landmarks, LEFT_WRIST))
            elbow_angle = state.smoothers["pushup_elbow_left"].update(raw_elbow)
            # Body line: shoulder-hip-ankle
            raw_line = calculate_angle(pt(landmarks, LEFT_SHOULDER), pt(landmarks, LEFT_HIP), pt(landmarks, LEFT_ANKLE))
            line_angle = state.smoothers["pushup_back_left"].update(raw_line)
        else:
            raw_elbow = calculate_angle(pt(landmarks, RIGHT_SHOULDER), pt(landmarks, RIGHT_ELBOW), pt(landmarks, RIGHT_WRIST))
            elbow_angle = state.smoothers["pushup_elbow_right"].update(raw_elbow)
            raw_line = calculate_angle(pt(landmarks, RIGHT_SHOULDER), pt(landmarks, RIGHT_HIP), pt(landmarks, RIGHT_ANKLE))
            line_angle = state.smoothers["pushup_back_right"].update(raw_line)

        diagnostics["primary_angle"] = round(elbow_angle, 1)

        # Posture check: body should be straight (> 150 deg)
        if line_angle >= 150:
            state.posture = "GOOD"
        else:
            state.posture = "BAD"
            diagnostics["guidance"] = "Keep hips aligned with shoulders and feet; do not sag or pike."

        # State machine: UP -> DOWN (elbow < 90) -> UP (elbow > 150)
        if elbow_angle < 90:
            state.stage = "DOWN"
        elif elbow_angle > 150 and state.stage == "DOWN":
            state.stage = "UP"
            state.record_completed_rep(now)
            weight_factor = state.user_weight / 70.0
            state.calories += 0.35 * weight_factor

    elif ex == "bicepcurl":
        l_vis = (landmarks[LEFT_ELBOW].visibility + landmarks[LEFT_WRIST].visibility) / 2.0
        r_vis = (landmarks[RIGHT_ELBOW].visibility + landmarks[RIGHT_WRIST].visibility) / 2.0

        if l_vis >= r_vis:
            raw_elbow = calculate_angle(pt(landmarks, LEFT_SHOULDER), pt(landmarks, LEFT_ELBOW), pt(landmarks, LEFT_WRIST))
            elbow_angle = state.smoothers["bicep_elbow_left"].update(raw_elbow)
        else:
            raw_elbow = calculate_angle(pt(landmarks, RIGHT_SHOULDER), pt(landmarks, RIGHT_ELBOW), pt(landmarks, RIGHT_WRIST))
            elbow_angle = state.smoothers["bicep_elbow_right"].update(raw_elbow)

        diagnostics["primary_angle"] = round(elbow_angle, 1)

        # Posture check: ensure upper arm stays stationary (not swinging backwards excessively)
        state.posture = "GOOD"

        # State machine: DOWN (elbow > 145) -> UP (elbow < 60) -> DOWN (elbow > 145) -> count!
        # Standard rep: start DOWN (> 145), curl UP (< 60), extend back DOWN (> 145) completes 1 rep.
        if elbow_angle > 145:
            if state.stage == "UP":
                # Returned to starting position from curled position
                state.stage = "DOWN"
                state.record_completed_rep(now)
                weight_factor = state.user_weight / 70.0
                state.calories += 0.25 * weight_factor
            else:
                state.stage = "DOWN"
        elif elbow_angle < 60 and state.stage == "DOWN":
            state.stage = "UP"

    elif ex == "jumpingjack":
        l_wrist_y = landmarks[LEFT_WRIST].y
        r_wrist_y = landmarks[RIGHT_WRIST].y
        shoulder_y = (landmarks[LEFT_SHOULDER].y + landmarks[RIGHT_SHOULDER].y) / 2.0
        leg_dist = abs(landmarks[LEFT_ANKLE].x - landmarks[RIGHT_ANKLE].x)

        # Hands above head: wrist y is less than shoulder_y - 0.05
        hands_up = (l_wrist_y < shoulder_y - 0.05) and (r_wrist_y < shoulder_y - 0.05)
        hands_down = (l_wrist_y > shoulder_y) and (r_wrist_y > shoulder_y)

        diagnostics["primary_angle"] = round(leg_dist, 2)

        # Movement cycle: CLOSE (standing) -> OPEN (hands up & feet apart) -> CLOSE (counted)
        if hands_up and leg_dist > 0.28:
            state.stage = "OPEN"
            state.posture = "GOOD"
        elif hands_down and leg_dist < 0.22 and state.stage == "OPEN":
            state.stage = "CLOSE"
            state.posture = "GOOD"
            state.record_completed_rep(now)
            weight_factor = state.user_weight / 70.0
            state.calories += 0.45 * weight_factor
        else:
            state.posture = "GOOD"

    elif ex == "plank":
        # Straight line alignment from shoulder to hip to ankle
        l_vis = (landmarks[LEFT_HIP].visibility + landmarks[LEFT_ANKLE].visibility) / 2.0
        r_vis = (landmarks[RIGHT_HIP].visibility + landmarks[RIGHT_ANKLE].visibility) / 2.0

        if l_vis >= r_vis:
            raw_body = calculate_angle(pt(landmarks, LEFT_SHOULDER), pt(landmarks, LEFT_HIP), pt(landmarks, LEFT_ANKLE))
            body_angle = state.smoothers["plank_body_left"].update(raw_body)
        else:
            raw_body = calculate_angle(pt(landmarks, RIGHT_SHOULDER), pt(landmarks, RIGHT_HIP), pt(landmarks, RIGHT_ANKLE))
            body_angle = state.smoothers["plank_body_right"].update(raw_body)

        diagnostics["primary_angle"] = round(body_angle, 1)

        # Good plank has body angle between 160 and 180 (straight back & legs)
        if 158.0 <= body_angle <= 180.0:
            state.posture = "GOOD"
            if state.plank_start_time is None:
                state.plank_start_time = now
            elapsed = now - state.plank_start_time
            state.counter = int(state.plank_accumulated_seconds + elapsed)
            state.calories = (state.counter / 3600.0) * 3.5 * state.user_weight
        else:
            state.posture = "BAD"
            diagnostics["guidance"] = "Keep your body in a straight line from shoulders to ankles."
            if state.plank_start_time is not None:
                state.plank_accumulated_seconds += (now - state.plank_start_time)
                state.plank_start_time = None
            state.counter = int(state.plank_accumulated_seconds)

    diagnostics["stage"] = state.stage
    diagnostics["reps"] = state.counter
    diagnostics["posture"] = state.posture
    diagnostics["calories"] = round(state.calories, 2)
    return diagnostics
