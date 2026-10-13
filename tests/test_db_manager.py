import unittest
from unittest.mock import MagicMock
from db_manager import DatabaseManager

class TestDatabaseManager(unittest.TestCase):

    def test_offline_mode_when_no_uri(self):
        db = DatabaseManager(mongo_uri="")
        self.assertFalse(db.is_connected)
        self.assertIsNone(db.get_user_profile("user@example.com"))
        self.assertEqual(db.get_user_workouts("user@example.com"), [])
        self.assertFalse(db.save_completed_workout("user@example.com", {"reps": 10}))

    def test_user_isolation_mocked(self):
        db = DatabaseManager(mongo_uri=None)
        # Mock database and collections
        db.is_connected = True
        mock_workouts = MagicMock()
        mock_profiles = MagicMock()
        db.workouts = mock_workouts
        db.profiles = mock_profiles

        # Test save_completed_workout
        db.save_completed_workout("userA@example.com", {
            "exercise": "squat",
            "reps": 15,
            "calories": 7.5,
            "posture": "GOOD",
            "summary": "Squats * 15"
        })

        mock_workouts.insert_one.assert_called_once()
        args, _ = mock_workouts.insert_one.call_args
        inserted_doc = args[0]
        self.assertEqual(inserted_doc["user_email"], "userA@example.com")
        self.assertEqual(inserted_doc["reps"], 15)

        # Test save_user_profile upserts into profiles collection only
        db.save_user_profile("userA@example.com", "User A", 75.0)
        mock_profiles.update_one.assert_called_once()
        filter_doc = mock_profiles.update_one.call_args[0][0]
        self.assertEqual(filter_doc["email"], "userA@example.com")

        # Test clear_user_workouts deletes ONLY userA workouts, does NOT delete entire collection
        db.clear_user_workouts("userA@example.com")
        mock_workouts.delete_many.assert_called_once_with({"user_email": "userA@example.com"})

if __name__ == "__main__":
    unittest.main()
