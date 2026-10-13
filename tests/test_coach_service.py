import unittest
from unittest.mock import MagicMock, patch
import os
from coach_service import ask_fitness_coach

class TestCoachService(unittest.TestCase):

    def test_empty_message(self):
        reply, code = ask_fitness_coach("")
        self.assertEqual(code, 400)
        self.assertIn("provide a question", reply)

    @patch.dict(os.environ, {"GROQ_API_KEY": ""})
    def test_missing_api_key(self):
        reply, code = ask_fitness_coach("How do I squat correctly?")
        self.assertEqual(code, 503)
        self.assertIn("GROQ_API_KEY", reply)

    @patch("coach_service.get_groq_client")
    @patch.dict(os.environ, {"GROQ_API_KEY": "dummy_key"})
    def test_mock_successful_response(self, mock_get_client):
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Keep your core tight and back straight during squats!"
        mock_choice.message.reasoning = None
        mock_client.chat.completions.create.return_value.choices = [mock_choice]
        mock_get_client.return_value = mock_client

        reply, code = ask_fitness_coach("Give me squat tips", {"exercise": "squat", "reps": 5})
        self.assertEqual(code, 200)
        self.assertIn("core tight", reply)

    @patch("coach_service.get_groq_client")
    @patch.dict(os.environ, {"GROQ_API_KEY": "invalid_key"})
    def test_mock_401_invalid_api_key(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("Error code: 401 - {'error': {'message': 'Invalid API Key'}}")
        mock_get_client.return_value = mock_client

        reply, code = ask_fitness_coach("Hello coach")
        self.assertEqual(code, 401)
        self.assertIn("invalid or unauthorized", reply)

    @patch("coach_service.get_groq_client")
    @patch.dict(os.environ, {"GROQ_API_KEY": "dummy_key"})
    def test_mock_429_rate_limit(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("Error code: 429 - Rate limit exceeded")
        mock_get_client.return_value = mock_client

        reply, code = ask_fitness_coach("Hello coach")
        self.assertEqual(code, 429)
        self.assertIn("high demand", reply)

if __name__ == "__main__":
    unittest.main()
