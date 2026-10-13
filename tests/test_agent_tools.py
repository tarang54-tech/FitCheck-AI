import unittest
from unittest.mock import MagicMock

from agent_tools import FitnessAgentTools, validate_plan
from exercise_tracker import WorkoutSessionState


class TestAgentTools(unittest.TestCase):
    def setUp(self):
        self.state = WorkoutSessionState(exercise="squat", user_weight=72)
        self.state.counter = 12
        self.db = MagicMock()
        self.db.is_connected = True
        self.tools = FitnessAgentTools("a@example.com", self.state, self.db, {"detected": True})

    def test_current_workout_uses_live_state(self):
        result, _ = self.tools.execute("get_current_workout", {})
        self.assertTrue(result["ok"])
        self.assertEqual(result["data"]["reps"], 12)
        self.assertTrue(result["data"]["tracking"]["detected"])

    def test_history_isolated_to_authenticated_user(self):
        self.db.get_user_workouts.return_value = []
        self.tools.execute("get_recent_workout_history", {"limit": 3})
        self.db.get_user_workouts.assert_called_once_with("a@example.com", limit=3)

    def test_history_reports_database_unavailable(self):
        self.db.is_connected = False
        result, _ = self.tools.execute("get_recent_workout_history", {})
        self.assertFalse(result["ok"])
        self.assertIn("database", result["error"])

    def test_progress_calculation_is_deterministic(self):
        self.db.get_user_workouts.return_value = [
            {"exercise": "squat", "reps": 10, "calories": 4, "timestamp": __import__("time").time()},
            {"exercise": "plank", "reps": 30, "calories": 3.5, "timestamp": __import__("time").time()},
        ]
        result, _ = self.tools.execute("analyze_workout_progress", {"period_days": 7})
        self.assertEqual(result["data"]["workout_count"], 2)
        self.assertEqual(result["data"]["repetitions_or_seconds_by_exercise"]["plank"], 30)
        self.assertEqual(result["data"]["estimated_calories"], 7.5)

    def test_plan_is_preview_only(self):
        result, plan = self.tools.execute("build_personalized_workout_plan", {"duration_minutes": 15, "experience": "beginner"})
        self.assertTrue(result["ok"])
        self.assertEqual(plan["duration_minutes"], 15)
        self.assertFalse(self.db.save_user_plan.called)

    def test_unknown_tool_is_rejected(self):
        result, _ = self.tools.execute("drop_database", {})
        self.assertFalse(result["ok"])

    def test_tool_rejects_unexpected_arguments(self):
        result, _ = self.tools.execute("get_current_workout", {"email": "other@example.com"})
        self.assertFalse(result["ok"])

    def test_plan_validation_rejects_unknown_exercise(self):
        plan, error = validate_plan({"duration_minutes": 20, "exercises": [{"exercise": "run", "sets": 2, "reps": 5}]})
        self.assertIsNone(plan)
        self.assertIn("unsupported", error)
