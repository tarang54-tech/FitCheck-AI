import os
import time
import json
import logging
from pymongo import MongoClient
from pymongo.errors import PyMongoError

logger = logging.getLogger("fitcheck.db")

class DatabaseManager:
    def __init__(self, mongo_uri=None):
        if mongo_uri is not None:
            self.mongo_uri = mongo_uri
        else:
            self.mongo_uri = os.getenv("MONGO_URI")
        self.client = None
        self.db = None
        self.workouts = None
        self.profiles = None
        self.plans = None
        self.is_connected = False
        self._init_connection()

    def close(self):
        if self.client:
            try:
                self.client.close()
            except Exception:
                pass

    def _init_connection(self):
        if not self.mongo_uri:
            logger.warning("MONGO_URI is not set. Database persistence will be disabled in offline mode.")
            self.is_connected = False
            return

        try:
            self.client = MongoClient(self.mongo_uri, serverSelectionTimeoutMS=4000)
            self.client.admin.command("ping")
            self.db = self.client["fitcheck"]
            # Two distinct collections to prevent any accidental deletion of user profiles
            self.workouts = self.db["workouts"]
            self.profiles = self.db["profiles"]
            self.plans = self.db["plans"]
            self.is_connected = True
            logger.info("MongoDB connected successfully.")
        except Exception as e:
            logger.error("MongoDB initial connection failed: %s", e)
            self.is_connected = False

    def ping(self):
        if not self.is_connected or self.client is None:
            return False
        try:
            self.client.admin.command("ping")
            return True
        except Exception:
            return False

    def get_user_profile(self, user_email):
        """Retrieve user profile safely without throwing unhandled exceptions."""
        if not self.is_connected or not user_email or self.profiles is None:
            return None
        try:
            return self.profiles.find_one({"email": user_email}, {"_id": 0})
        except Exception as e:
            logger.error("Error fetching user profile: %s", e)
            return None

    def save_user_profile(self, user_email, name, weight):
        """Atomic upsert for user profile."""
        if not self.is_connected or not user_email or self.profiles is None:
            return False
        try:
            self.profiles.update_one(
                {"email": user_email},
                {"$set": {
                    "email": user_email,
                    "name": name,
                    "weight": float(weight),
                    "updated_at": time.time()
                }},
                upsert=True
            )
            return True
        except Exception as e:
            logger.error("Error saving user profile: %s", e)
            return False

    def save_completed_workout(self, user_email, session_data):
        """
        Saves a single completed workout document for a specific user.
        NEVER drops or clears other users' data.
        """
        if not self.is_connected or not user_email or self.workouts is None:
            return False
        try:
            doc = {
                "user_email": user_email,
                "exercise": session_data.get("exercise"),
                "reps": session_data.get("reps", 0),
                "calories": session_data.get("calories", 0.0),
                "posture": session_data.get("posture", "GOOD"),
                "summary": session_data.get("summary", ""),
                "timestamp": session_data.get("timestamp", time.time())
            }
            self.workouts.insert_one(doc)
            return True
        except Exception as e:
            logger.error("Error inserting workout document: %s", e)
            return False

    def get_user_workouts(self, user_email, limit=30):
        """Fetches workout history specifically filtered by user email."""
        if not self.is_connected or not user_email or self.workouts is None:
            return []
        try:
            cursor = self.workouts.find(
                {"user_email": user_email},
                {"_id": 0}
            ).sort("timestamp", -1).limit(limit)
            return list(cursor)
        except Exception as e:
            logger.error("Error fetching user workouts: %s", e)
            return []

    def clear_user_workouts(self, user_email):
        """Safely clears workouts ONLY for the specific requesting user."""
        if not self.is_connected or not user_email or self.workouts is None:
            return False
        try:
            self.workouts.delete_many({"user_email": user_email})
            return True
        except Exception as e:
            logger.error("Error clearing user workouts: %s", e)
            return False

    def save_user_plan(self, user_email, plan):
        """Save a validated plan for one user without touching workouts or profiles."""
        if not self.is_connected or not user_email or self.plans is None or not isinstance(plan, dict):
            return False
        try:
            self.plans.insert_one({
                "user_email": user_email,
                "title": plan["title"],
                "duration_minutes": plan["duration_minutes"],
                "exercises": plan["exercises"],
                "created_at": time.time(),
            })
            return True
        except Exception as e:
            logger.error("Error saving workout plan: %s", e)
            return False

    def get_user_plans(self, user_email, limit=10):
        """Read saved plans belonging only to the authenticated user."""
        if not self.is_connected or not user_email or self.plans is None:
            return None
        try:
            cursor = self.plans.find({"user_email": user_email}, {"_id": 0, "user_email": 0}).sort("created_at", -1).limit(min(max(int(limit), 1), 20))
            return list(cursor)
        except Exception as e:
            logger.error("Error fetching workout plans: %s", e)
            return None
