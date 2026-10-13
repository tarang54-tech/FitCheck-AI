import unittest
from unittest.mock import MagicMock, patch, ANY
import json
import os

from app import app, user_sessions

class TestAppRoutes(unittest.TestCase):

    def setUp(self):
        app.config["TESTING"] = True
        app.config["SECRET_KEY"] = "test-secret-key"
        self.client = app.test_client()

    def test_unauthenticated_access_redirects_or_rejects(self):
        # Index redirects to login
        res = self.client.get("/")
        self.assertEqual(res.status_code, 302)
        self.assertIn("/login", res.headers["Location"])

        # Frame processing rejects unauthenticated calls
        res = self.client.post("/process_frame", data=b"empty")
        self.assertEqual(res.status_code, 401)

        # Chat rejects unauthenticated calls
        res = self.client.post("/chat", json={"message": "hello"})
        self.assertEqual(res.status_code, 401)

        # Stats rejects unauthenticated calls
        res = self.client.get("/stats")
        self.assertEqual(res.status_code, 401)

    def test_authenticated_set_weight_and_stats(self):
        with self.client.session_transaction() as sess:
            sess["user"] = {
                "name": "Jane Tester",
                "email": "jane@example.com"
            }

        # Set valid weight
        res = self.client.post("/set_weight", json={"weight": 65})
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertTrue(data.get("ok"))
        self.assertEqual(data.get("weight"), 65.0)

        # Set invalid weight
        res = self.client.post("/set_weight", json={"weight": 350})
        self.assertEqual(res.status_code, 400)

        # Stats retrieves updated state
        res = self.client.get("/stats")
        self.assertEqual(res.status_code, 200)
        stats_data = json.loads(res.data)
        self.assertEqual(stats_data.get("user_weight"), 65.0)
        self.assertEqual(stats_data.get("exercise"), "squat")

    def test_exercise_switching(self):
        with self.client.session_transaction() as sess:
            sess["user"] = {
                "name": "Alex Fit",
                "email": "alex@example.com"
            }

        res = self.client.post("/set_exercise/bicepcurl")
        self.assertEqual(res.status_code, 200)

        res = self.client.get("/stats")
        stats_data = json.loads(res.data)
        self.assertEqual(stats_data.get("exercise"), "bicepcurl")
        self.assertEqual(stats_data.get("stage"), "DOWN")

        # Invalid exercise
        res = self.client.post("/set_exercise/unknown")
        self.assertEqual(res.status_code, 400)

    def test_user_session_isolation(self):
        # User 1 sets pushup
        client1 = app.test_client()
        with client1.session_transaction() as sess:
            sess["user"] = {"name": "User 1", "email": "user1@example.com"}
        client1.post("/set_exercise/pushup")

        # User 2 sets bicepcurl
        client2 = app.test_client()
        with client2.session_transaction() as sess:
            sess["user"] = {"name": "User 2", "email": "user2@example.com"}
        client2.post("/set_exercise/bicepcurl")

        # User 1 stats should remain pushup
        res1 = client1.get("/stats")
        self.assertEqual(json.loads(res1.data)["exercise"], "pushup")

        # User 2 stats should remain bicepcurl
        res2 = client2.get("/stats")
        self.assertEqual(json.loads(res2.data)["exercise"], "bicepcurl")

    def test_frame_processing_invalid_payload(self):
        with self.client.session_transaction() as sess:
            sess["user"] = {"name": "Test User", "email": "test@example.com"}

        # Empty body
        res = self.client.post("/process_frame", data=b"")
        self.assertEqual(res.status_code, 400)

        # Corrupt body
        res = self.client.post("/process_frame", data=b"not-an-image")
        self.assertEqual(res.status_code, 400)

    @patch("app.run_fitness_agent")
    def test_chat_delegation(self, mock_agent):
        mock_agent.return_value = {"reply": "Keep your back straight!", "status": 200, "tool_activity": ["get_current_workout"], "plan": None, "proposal": None}

        with self.client.session_transaction() as sess:
            sess["user"] = {"name": "Test User", "email": "test@example.com"}

        res = self.client.post("/chat", json={"message": "Help with squat"})
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertIn("Keep your back straight!", data["reply"])
        self.assertEqual(data["tool_activity"], ["get_current_workout"])

    def test_plan_endpoints_require_login(self):
        self.assertEqual(self.client.get("/plans").status_code, 401)
        self.assertEqual(self.client.post("/plans", json={"plan": {}}).status_code, 401)

    @patch("app.db_manager")
    def test_plan_saves_only_after_explicit_post(self, mock_db):
        mock_db.is_connected = True
        mock_db.save_user_plan.return_value = True
        with self.client.session_transaction() as sess:
            sess["user"] = {"name": "Plan User", "email": "plans@example.com"}
        plan = {"title": "Quick plan", "duration_minutes": 15, "exercises": [{"exercise": "squat", "sets": 2, "reps": 8, "rest_seconds": 60}]}
        response = self.client.post("/plans", json={"plan": plan})
        self.assertEqual(response.status_code, 201)
        mock_db.save_user_plan.assert_called_once_with("plans@example.com", ANY)

    @patch("app.db_manager")
    def test_plan_retrieval_is_authorized_and_user_scoped(self, mock_db):
        mock_db.get_user_plans.return_value = [{"title": "My plan", "duration_minutes": 15, "exercises": []}]
        with self.client.session_transaction() as sess:
            sess["user"] = {"name": "Plan User", "email": "plans@example.com"}
        response = self.client.get("/plans")
        self.assertEqual(response.status_code, 200)
        mock_db.get_user_plans.assert_called_once_with("plans@example.com", limit=10)

    def test_health_check(self):
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertEqual(data["status"], "healthy")

if __name__ == "__main__":
    unittest.main()
