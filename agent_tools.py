"""Narrow, server-side tools available to the FitCheck AI coach."""
import time

SUPPORTED_EXERCISES = {"squat", "pushup", "jumpingjack", "bicepcurl", "plank"}
MAX_HISTORY = 20
TOOL_ARGUMENTS = {
    "get_current_workout": set(),
    "get_recent_workout_history": {"limit"},
    "get_user_profile": set(),
    "analyze_workout_progress": {"period_days"},
    "build_personalized_workout_plan": {"duration_minutes", "experience", "equipment"},
    "propose_exercise_change": {"exercise", "reason"},
}


TOOL_SCHEMAS = [
    {"type": "function", "function": {"name": "get_current_workout", "description": "Read the authenticated user's current live workout state.", "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "get_recent_workout_history", "description": "Read saved workouts for the authenticated user only.", "parameters": {"type": "object", "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 20}}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "get_user_profile", "description": "Read safe stored profile data for the authenticated user.", "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "analyze_workout_progress", "description": "Calculate deterministic progress statistics from saved workouts.", "parameters": {"type": "object", "properties": {"period_days": {"type": "integer", "minimum": 1, "maximum": 90}}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "build_personalized_workout_plan", "description": "Build a conservative draft plan using only FitCheck-supported exercises. This does not save or change anything.", "parameters": {"type": "object", "properties": {"duration_minutes": {"type": "integer", "minimum": 5, "maximum": 120}, "experience": {"type": "string", "enum": ["beginner", "intermediate", "advanced"]}, "equipment": {"type": "string", "maxLength": 100}}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "propose_exercise_change", "description": "Propose, but never apply, a change to one supported active exercise.", "parameters": {"type": "object", "properties": {"exercise": {"type": "string", "enum": sorted(SUPPORTED_EXERCISES)}, "reason": {"type": "string", "maxLength": 240}}, "required": ["exercise", "reason"], "additionalProperties": False}}},
]


def validate_plan(plan):
    """Normalize a draft plan before it can be previewed or persisted."""
    if not isinstance(plan, dict):
        return None, "Plan must be an object."
    title = str(plan.get("title", "Personalized FitCheck plan")).strip()[:80]
    duration = plan.get("duration_minutes")
    exercises = plan.get("exercises")
    if not isinstance(duration, int) or not 5 <= duration <= 120:
        return None, "Plan duration must be between 5 and 120 minutes."
    if not isinstance(exercises, list) or not 1 <= len(exercises) <= 5:
        return None, "Plan must contain between one and five exercises."
    cleaned = []
    for item in exercises:
        if not isinstance(item, dict) or item.get("exercise") not in SUPPORTED_EXERCISES:
            return None, "Plan contains an unsupported exercise."
        cleaned_item = {"exercise": item["exercise"]}
        for key, minimum, maximum in (("sets", 1, 10), ("reps", 1, 100), ("duration_seconds", 10, 3600), ("rest_seconds", 0, 600)):
            value = item.get(key)
            if value is not None:
                if not isinstance(value, int) or not minimum <= value <= maximum:
                    return None, f"Invalid {key} in plan."
                cleaned_item[key] = value
        if "reps" not in cleaned_item and "duration_seconds" not in cleaned_item:
            return None, "Each exercise needs repetitions or a duration."
        reminder = str(item.get("reminder", "Move with control and stop for pain.")).strip()[:240]
        cleaned_item["reminder"] = reminder
        cleaned.append(cleaned_item)
    return {"title": title or "Personalized FitCheck plan", "duration_minutes": duration, "exercises": cleaned}, None


class FitnessAgentTools:
    """Fixed allowlist; every method is bound to one authenticated user."""
    def __init__(self, user_email, user_state, db_manager, tracking_status=None):
        self.user_email = user_email
        self.user_state = user_state
        self.db_manager = db_manager
        self.tracking_status = tracking_status or {}

    def execute(self, name, arguments):
        if name not in TOOL_ARGUMENTS:
            return {"ok": False, "error": "Unknown tool."}, None
        if not isinstance(arguments, dict):
            return {"ok": False, "error": "Tool arguments must be a JSON object."}, None
        if set(arguments) - TOOL_ARGUMENTS[name]:
            return {"ok": False, "error": "Tool arguments contain unsupported fields."}, None
        return getattr(self, name)(arguments)

    def _history(self, limit=MAX_HISTORY):
        if not self.db_manager.is_connected:
            return None
        return self.db_manager.get_user_workouts(self.user_email, limit=min(max(limit, 1), MAX_HISTORY))

    def get_current_workout(self, _args):
        state = self.user_state.to_dict()
        return {"ok": True, "data": {**state, "tracking": self.tracking_status}}, None

    def get_recent_workout_history(self, args):
        limit = args.get("limit", 10)
        if not isinstance(limit, int):
            return {"ok": False, "error": "limit must be an integer."}, None
        history = self._history(limit)
        if history is None:
            return {"ok": False, "error": "Workout history is unavailable because the database is not connected."}, None
        if not history:
            return {"ok": True, "data": [], "message": "No workouts recorded yet."}, None
        return {"ok": True, "data": history}, None

    def get_user_profile(self, _args):
        profile = self.db_manager.get_user_profile(self.user_email) if self.db_manager.is_connected else None
        data = {"weight": self.user_state.user_weight}
        if profile:
            data.update({key: profile[key] for key in ("name", "weight") if key in profile})
        return {"ok": True, "data": data, "source": "profile" if profile else "current_session"}, None

    def analyze_workout_progress(self, args):
        days = args.get("period_days", 30)
        if not isinstance(days, int) or not 1 <= days <= 90:
            return {"ok": False, "error": "period_days must be an integer from 1 to 90."}, None
        history = self._history(MAX_HISTORY)
        if history is None:
            return {"ok": False, "error": "Progress cannot be calculated because the database is unavailable."}, None
        cutoff = time.time() - days * 86400
        sessions = [item for item in history if isinstance(item.get("timestamp"), (int, float)) and item["timestamp"] >= cutoff]
        totals, calories = {}, 0.0
        for item in sessions:
            exercise = item.get("exercise", "unknown")
            totals[exercise] = totals.get(exercise, 0) + int(item.get("reps", 0) or 0)
            calories += float(item.get("calories", 0) or 0)
        return {"ok": True, "data": {"period_days": days, "workout_count": len(sessions), "frequency_per_week": round(len(sessions) / (days / 7), 1), "repetitions_or_seconds_by_exercise": totals, "estimated_calories": round(calories, 1), "enough_data": len(sessions) >= 2}}, None

    def build_personalized_workout_plan(self, args):
        duration = args.get("duration_minutes")
        if not isinstance(duration, int) or not 5 <= duration <= 120:
            return {"ok": False, "error": "Ask the user for a workout duration between 5 and 120 minutes before building a plan."}, None
        experience = args.get("experience", "beginner")
        if experience not in {"beginner", "intermediate", "advanced"}:
            return {"ok": False, "error": "Experience must be beginner, intermediate, or advanced."}, None
        base_reps = {"beginner": 8, "intermediate": 12, "advanced": 16}[experience]
        exercises = [
            {"exercise": "squat", "sets": 2 if duration < 20 else 3, "reps": base_reps, "rest_seconds": 60, "reminder": "Keep knees tracking over toes."},
            {"exercise": "pushup", "sets": 2 if duration < 20 else 3, "reps": max(5, base_reps - 2), "rest_seconds": 75, "reminder": "Keep your body line steady."},
            {"exercise": "plank", "sets": 2, "duration_seconds": 20 if experience == "beginner" else 35, "rest_seconds": 45, "reminder": "Keep shoulders, hips, and ankles aligned."},
        ]
        plan, error = validate_plan({"title": f"{duration}-minute FitCheck plan", "duration_minutes": duration, "exercises": exercises})
        return ({"ok": True, "data": plan, "message": "Draft only; user confirmation is required before saving."}, plan) if not error else ({"ok": False, "error": error}, None)

    def propose_exercise_change(self, args):
        exercise, reason = args.get("exercise"), args.get("reason")
        if exercise not in SUPPORTED_EXERCISES or not isinstance(reason, str) or not reason.strip():
            return {"ok": False, "error": "A supported exercise and concise reason are required."}, None
        return {"ok": True, "proposal": {"exercise": exercise, "reason": reason.strip()[:240]}, "message": "Proposal only; the user must confirm any change in the dashboard."}, None
