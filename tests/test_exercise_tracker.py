import unittest
from unittest.mock import MagicMock
from exercise_tracker import (
    calculate_angle,
    AngleSmoother,
    WorkoutSessionState,
    evaluate_frame_pose,
    check_visibility,
    EXERCISE_LANDMARKS
)

class MockLandmark:
    def __init__(self, x, y, z=0.0, visibility=0.99):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility

def create_mock_body(knee_angle_deg=170, elbow_angle_deg=160, back_straight=True, wrist_above_head=False, leg_spread=0.2):
    # Default 33 landmarks for MediaPipe pose
    landmarks = [MockLandmark(0.5, 0.5, visibility=0.95) for _ in range(33)]

    # Hips, knees, ankles on left & right
    # Hip at (0.5, 0.5)
    landmarks[23] = MockLandmark(0.5, 0.5, visibility=0.95)
    landmarks[24] = MockLandmark(0.5, 0.5, visibility=0.95)

    # Knee below hip
    landmarks[25] = MockLandmark(0.5, 0.7, visibility=0.95)
    landmarks[26] = MockLandmark(0.5, 0.7, visibility=0.95)

    # Calculate ankle position to reflect knee_angle_deg
    import math
    rad = math.radians(180 - knee_angle_deg)
    ankle_x = 0.5 + 0.2 * math.sin(rad)
    ankle_y = 0.7 + 0.2 * math.cos(rad)
    landmarks[27] = MockLandmark(ankle_x, ankle_y, visibility=0.95)
    landmarks[28] = MockLandmark(ankle_x + leg_spread, ankle_y, visibility=0.95)

    # Shoulder
    if back_straight:
        landmarks[11] = MockLandmark(0.5, 0.25, visibility=0.95)
        landmarks[12] = MockLandmark(0.5, 0.25, visibility=0.95)
    else:
        # Lean far forward
        landmarks[11] = MockLandmark(0.8, 0.4, visibility=0.95)
        landmarks[12] = MockLandmark(0.8, 0.4, visibility=0.95)

    # Elbow and wrist for pushups / bicep curls
    landmarks[13] = MockLandmark(0.5, 0.4, visibility=0.95)
    landmarks[14] = MockLandmark(0.5, 0.4, visibility=0.95)

    elbow_rad = math.radians(180 - elbow_angle_deg)
    wrist_x = 0.5 + 0.15 * math.sin(elbow_rad)
    wrist_y = 0.4 + ( -0.15 * math.cos(elbow_rad) if wrist_above_head else 0.15 * math.cos(elbow_rad) )
    landmarks[15] = MockLandmark(wrist_x, wrist_y, visibility=0.95)
    landmarks[16] = MockLandmark(wrist_x, wrist_y, visibility=0.95)

    return landmarks

class TestExerciseTracker(unittest.TestCase):

    def test_calculate_angle(self):
        # Right angle
        p1 = [0, 1]
        p2 = [0, 0]
        p3 = [1, 0]
        self.assertAlmostEqual(calculate_angle(p1, p2, p3), 90.0, places=2)

        # Straight line
        p1 = [-1, 0]
        p2 = [0, 0]
        p3 = [1, 0]
        self.assertAlmostEqual(calculate_angle(p1, p2, p3), 180.0, places=2)

        # Zero length points
        self.assertEqual(calculate_angle([0, 0], [0, 0], [1, 0]), 0.0)

    def test_angle_smoother_independent(self):
        s1 = AngleSmoother(window_size=3)
        s2 = AngleSmoother(window_size=3)

        s1.update(100)
        s1.update(110)
        self.assertEqual(s1.update(120), 110.0)

        # s2 must not be affected by s1
        self.assertIsNone(s2.value)
        self.assertEqual(s2.update(50), 50.0)

        # Reset s1
        s1.reset()
        self.assertIsNone(s1.value)
        self.assertEqual(s2.value, 50.0)

    def test_squat_rep_cycle(self):
        state = WorkoutSessionState(exercise="squat", user_weight=70.0)
        state.reset_exercise_tracking("squat")

        # 1. Standing position (> 155)
        lms_standing = create_mock_body(knee_angle_deg=170)
        diag = evaluate_frame_pose(state, lms_standing, now=100.0)
        self.assertEqual(state.counter, 0)
        self.assertEqual(state.stage, "UP")

        # 2. Squat down (< 110)
        lms_down = create_mock_body(knee_angle_deg=95)
        for _ in range(5):  # smooth into 95
            diag = evaluate_frame_pose(state, lms_down, now=101.0)
        self.assertEqual(state.stage, "DOWN")
        self.assertEqual(state.counter, 0)

        # 3. Stand back up (> 155)
        for _ in range(5):
            diag = evaluate_frame_pose(state, lms_standing, now=102.0)
        self.assertEqual(state.stage, "UP")
        self.assertEqual(state.counter, 1)
        self.assertGreater(state.calories, 0.0)

    def test_pushup_rep_cycle(self):
        state = WorkoutSessionState(exercise="pushup", user_weight=75.0)
        state.reset_exercise_tracking("pushup")

        lms_up = create_mock_body(elbow_angle_deg=165)
        for _ in range(5):
            evaluate_frame_pose(state, lms_up, now=100.0)
        self.assertEqual(state.stage, "UP")
        self.assertEqual(state.counter, 0)

        lms_down = create_mock_body(elbow_angle_deg=80)
        for _ in range(5):
            evaluate_frame_pose(state, lms_down, now=101.0)
        self.assertEqual(state.stage, "DOWN")
        self.assertEqual(state.counter, 0)

        for _ in range(5):
            evaluate_frame_pose(state, lms_up, now=102.0)
        self.assertEqual(state.stage, "UP")
        self.assertEqual(state.counter, 1)

    def test_bicep_curl_rep_cycle(self):
        state = WorkoutSessionState(exercise="bicepcurl", user_weight=70.0)
        state.reset_exercise_tracking("bicepcurl")

        # Start down
        lms_down = create_mock_body(elbow_angle_deg=160)
        for _ in range(5):
            evaluate_frame_pose(state, lms_down, now=100.0)
        self.assertEqual(state.stage, "DOWN")
        self.assertEqual(state.counter, 0)

        # Curl up (< 60)
        lms_up = create_mock_body(elbow_angle_deg=45)
        for _ in range(5):
            evaluate_frame_pose(state, lms_up, now=101.0)
        self.assertEqual(state.stage, "UP")
        self.assertEqual(state.counter, 0)

        # Extend down (> 145) completes the rep
        for _ in range(5):
            evaluate_frame_pose(state, lms_down, now=102.0)
        self.assertEqual(state.stage, "DOWN")
        self.assertEqual(state.counter, 1)

    def test_jumping_jack_rep_cycle(self):
        state = WorkoutSessionState(exercise="jumpingjack", user_weight=70.0)
        state.reset_exercise_tracking("jumpingjack")

        # Initial close
        lms_close = [MockLandmark(0.5, 0.5, visibility=0.95) for _ in range(33)]
        lms_close[11] = MockLandmark(0.45, 0.4, visibility=0.95)
        lms_close[12] = MockLandmark(0.55, 0.4, visibility=0.95)
        lms_close[15] = MockLandmark(0.4, 0.6, visibility=0.95)  # wrists down
        lms_close[16] = MockLandmark(0.6, 0.6, visibility=0.95)
        lms_close[27] = MockLandmark(0.45, 0.9, visibility=0.95) # ankles close
        lms_close[28] = MockLandmark(0.55, 0.9, visibility=0.95)

        evaluate_frame_pose(state, lms_close, now=100.0)
        self.assertEqual(state.counter, 0)

        # Open: hands up and legs wide
        lms_open = [MockLandmark(0.5, 0.5, visibility=0.95) for _ in range(33)]
        lms_open[11] = MockLandmark(0.45, 0.4, visibility=0.95)
        lms_open[12] = MockLandmark(0.55, 0.4, visibility=0.95)
        lms_open[15] = MockLandmark(0.4, 0.2, visibility=0.95)  # wrists up
        lms_open[16] = MockLandmark(0.6, 0.2, visibility=0.95)
        lms_open[27] = MockLandmark(0.3, 0.9, visibility=0.95)  # legs wide (> 0.28)
        lms_open[28] = MockLandmark(0.7, 0.9, visibility=0.95)

        evaluate_frame_pose(state, lms_open, now=101.0)
        self.assertEqual(state.stage, "OPEN")
        self.assertEqual(state.counter, 0)

        # Back to close
        evaluate_frame_pose(state, lms_close, now=102.0)
        self.assertEqual(state.stage, "CLOSE")
        self.assertEqual(state.counter, 1)

    def test_plank_timer_and_posture_loss(self):
        state = WorkoutSessionState(exercise="plank", user_weight=70.0)
        state.reset_exercise_tracking("plank")

        # Straight body line: shoulder(0.2, 0.5), hip(0.5, 0.5), ankle(0.8, 0.5)
        lms_good = [MockLandmark(0.5, 0.5, visibility=0.95) for _ in range(33)]
        lms_good[11] = MockLandmark(0.2, 0.5, visibility=0.95)
        lms_good[12] = MockLandmark(0.2, 0.5, visibility=0.95)
        lms_good[23] = MockLandmark(0.5, 0.5, visibility=0.95)
        lms_good[24] = MockLandmark(0.5, 0.5, visibility=0.95)
        lms_good[27] = MockLandmark(0.8, 0.5, visibility=0.95)
        lms_good[28] = MockLandmark(0.8, 0.5, visibility=0.95)

        for _ in range(5):
            evaluate_frame_pose(state, lms_good, now=100.0)
        self.assertEqual(state.posture, "GOOD")

        # Advance 5 seconds with good posture
        evaluate_frame_pose(state, lms_good, now=105.0)
        self.assertEqual(state.counter, 5)

        # Sagging hips (bad posture)
        lms_bad = [MockLandmark(0.5, 0.5, visibility=0.95) for _ in range(33)]
        lms_bad[11] = MockLandmark(0.2, 0.5, visibility=0.95)
        lms_bad[12] = MockLandmark(0.2, 0.5, visibility=0.95)
        lms_bad[23] = MockLandmark(0.5, 0.75, visibility=0.95) # sagging hip
        lms_bad[24] = MockLandmark(0.5, 0.75, visibility=0.95)
        lms_bad[27] = MockLandmark(0.8, 0.5, visibility=0.95)
        lms_bad[28] = MockLandmark(0.8, 0.5, visibility=0.95)

        # Posture lost after 5 seconds at now=105.0
        for _ in range(5):
            evaluate_frame_pose(state, lms_bad, now=105.0)
        self.assertEqual(state.posture, "BAD")
        self.assertIsNone(state.plank_start_time)

        # Ensure timer didn't run further during bad posture
        evaluate_frame_pose(state, lms_bad, now=110.0)
        self.assertEqual(state.counter, 5)

    def test_visibility_failure_gives_guidance(self):
        state = WorkoutSessionState(exercise="squat", user_weight=70.0)
        lms = [MockLandmark(0.5, 0.5, visibility=0.1) for _ in range(33)]
        diag = evaluate_frame_pose(state, lms, now=100.0)
        self.assertFalse(diag["visibility_ok"])
        self.assertIsNotNone(diag["guidance"])
        self.assertEqual(diag["posture"], "NOT_EVALUATED")

    def test_exercise_reset(self):
        state = WorkoutSessionState(exercise="squat", user_weight=70.0)
        state.counter = 10
        state.calories = 15.0
        state.stage = "DOWN"
        state.reset_exercise_tracking("bicepcurl")
        self.assertEqual(state.exercise, "bicepcurl")
        self.assertEqual(state.counter, 0)
        self.assertEqual(state.stage, "DOWN")
        self.assertEqual(state.calories, 0.0)

if __name__ == "__main__":
    unittest.main()
