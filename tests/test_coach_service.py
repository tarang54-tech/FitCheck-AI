import unittest
from unittest.mock import MagicMock, patch
import os
from coach_service import ask_fitness_coach, check_groq_credentials, run_fitness_agent


class MockGroqError(Exception):
    def __init__(self, status_code, message="Groq request failed"):
        super().__init__(message)
        self.status_code = status_code


class ToolCall:
    def __init__(self, call_id, name, arguments):
        self.id = call_id
        self.function = MagicMock(name=name, arguments=arguments)


class FakeTools:
    def __init__(self):
        self.calls = []
    def execute(self, name, arguments):
        self.calls.append((name, arguments))
        if name == "unknown":
            return {"ok": False, "error": "Unknown tool."}, None
        return {"ok": True, "data": {"reps": 7}}, None

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

    @patch.dict(os.environ, {"GROQ_API_KEY": "   "})
    def test_whitespace_only_key_is_missing(self):
        reply, code = ask_fitness_coach("How do I squat correctly?")
        self.assertEqual(code, 503)

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
        self.assertEqual(mock_client.chat.completions.create.call_count, 1)

    @patch("coach_service.get_groq_client")
    @patch.dict(os.environ, {"GROQ_API_KEY": "dummy_key"})
    def test_mock_429_rate_limit(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("Error code: 429 - Rate limit exceeded")
        mock_get_client.return_value = mock_client

        reply, code = ask_fitness_coach("Hello coach")
        self.assertEqual(code, 429)
        self.assertIn("high demand", reply)

    @patch("coach_service.get_groq_client")
    @patch.dict(os.environ, {"GROQ_API_KEY": "dummy_key"})
    def test_unsupported_model_uses_fallback(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            MockGroqError(404, "model not found"),
            MagicMock(choices=[MagicMock(message=MagicMock(content="Try controlled movement.", reasoning=None))]),
        ]
        mock_get_client.return_value = mock_client

        reply, code = ask_fitness_coach("How should I curl?", model=None)
        self.assertEqual(code, 200)
        self.assertIn("controlled movement", reply)
        self.assertEqual(mock_client.chat.completions.create.call_count, 2)

    @patch("coach_service.get_groq_client")
    @patch.dict(os.environ, {"GROQ_API_KEY": "dummy_key"})
    def test_timeout_does_not_retry_all_models(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = MockGroqError(504, "upstream timeout")
        mock_get_client.return_value = mock_client

        reply, code = ask_fitness_coach("Hello coach")
        self.assertEqual(code, 503)
        self.assertEqual(mock_client.chat.completions.create.call_count, 1)

    @patch("coach_service.get_groq_client")
    @patch.dict(os.environ, {"GROQ_API_KEY": "dummy_key"})
    def test_safe_credential_check_reports_rejected_key(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.models.list.side_effect = MockGroqError(401, "Invalid API Key")
        mock_get_client.return_value = mock_client

        result = check_groq_credentials()
        self.assertEqual(result, {
            "configured": True,
            "accepted": False,
            "status": "rejected",
            "http_status": 401,
        })

    @patch("coach_service.get_groq_client")
    @patch.dict(os.environ, {"GROQ_API_KEY": "dummy_key"})
    def test_safe_credential_check_reports_accepted_key(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        result = check_groq_credentials()
        self.assertEqual(result, {"configured": True, "accepted": True, "status": "accepted"})

    @patch("coach_service.get_groq_client")
    @patch.dict(os.environ, {"GROQ_API_KEY": "dummy_key"})
    def test_agent_direct_answer_without_tools(self, mock_get_client):
        client = MagicMock()
        client.chat.completions.create.return_value.choices = [MagicMock(message=MagicMock(content="Keep a neutral spine.", tool_calls=[]))]
        mock_get_client.return_value = client
        result = run_fitness_agent("How do I squat?", FakeTools())
        self.assertEqual(result["status"], 200)
        self.assertEqual(result["tool_activity"], [])

    @patch("coach_service.get_groq_client")
    @patch.dict(os.environ, {"GROQ_API_KEY": "dummy_key"})
    def test_agent_returns_tool_results_for_multi_step_calling(self, mock_get_client):
        client = MagicMock()
        first = MagicMock(content=None, tool_calls=[ToolCall("one", "get_current_workout", "{}")])
        second = MagicMock(content=None, tool_calls=[ToolCall("two", "analyze_workout_progress", '{"period_days":7}')])
        third = MagicMock(content="You completed 7 reps.", tool_calls=[])
        client.chat.completions.create.side_effect = [
            MagicMock(choices=[MagicMock(message=first)]), MagicMock(choices=[MagicMock(message=second)]), MagicMock(choices=[MagicMock(message=third)]),
        ]
        mock_get_client.return_value = client
        tools = FakeTools()
        result = run_fitness_agent("How am I doing?", tools)
        self.assertEqual(result["status"], 200)
        self.assertEqual(len(tools.calls), 2)
        third_messages = client.chat.completions.create.call_args_list[2].kwargs["messages"]
        self.assertEqual([m["role"] for m in third_messages if isinstance(m, dict)][-1], "tool")

    @patch("coach_service.get_groq_client")
    @patch.dict(os.environ, {"GROQ_API_KEY": "dummy_key"})
    def test_agent_rejects_unknown_and_malformed_tool_calls(self, mock_get_client):
        client = MagicMock()
        first = MagicMock(content=None, tool_calls=[ToolCall("one", "unknown", "not-json")])
        second = MagicMock(content="I could not access that data.", tool_calls=[])
        client.chat.completions.create.side_effect = [MagicMock(choices=[MagicMock(message=first)]), MagicMock(choices=[MagicMock(message=second)])]
        mock_get_client.return_value = client
        result = run_fitness_agent("Do a thing", FakeTools())
        self.assertEqual(result["status"], 200)
        self.assertIn("unavailable", result["tool_activity"][0])

    @patch("coach_service.get_groq_client")
    @patch.dict(os.environ, {"GROQ_API_KEY": "dummy_key"})
    def test_agent_enforces_tool_round_limit(self, mock_get_client):
        client = MagicMock()
        message = MagicMock(content=None, tool_calls=[ToolCall("one", "get_current_workout", "{}")])
        client.chat.completions.create.return_value.choices = [MagicMock(message=message)]
        mock_get_client.return_value = client
        result = run_fitness_agent("Keep checking", FakeTools())
        self.assertEqual(result["status"], 429)

if __name__ == "__main__":
    unittest.main()
