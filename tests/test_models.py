import json
import unittest
from datetime import datetime, timezone

from models import DecisionLog


def _row(observation=None, action=None, reward=-9.6, reason="exploit"):
    observation = observation or {
        "air_temp_c": 31.2,
        "humidity_pct": 40.0,
        "soc_pct": 55.0,
        "solar_power_w": 150.0,
        "forecast_high_c": 33.0,
        "timestamp": "2026-05-29T12:00:00+00:00",
    }
    action = action if action is not None else {"fan": {"percentage": 25}}
    return DecisionLog(
        id=7,
        timestamp=datetime(2026, 5, 29, 12, 0, tzinfo=timezone.utc),
        observation_json=json.dumps(observation),
        action_json=json.dumps(action),
        policy_name="q_learning_v1",
        reason=reason,
        reward=reward,
    )


class TestDecisionLogToApiDict(unittest.TestCase):
    def test_flattens_observation_fields(self):
        d = _row().to_api_dict()
        self.assertEqual(d["id"], 7)
        self.assertEqual(d["air_temp_c"], 31.2)
        self.assertEqual(d["soc_pct"], 55.0)
        self.assertEqual(d["solar_power_w"], 150.0)
        self.assertEqual(d["fan_percentage"], 25)
        self.assertEqual(d["reward"], -9.6)
        self.assertEqual(d["reason"], "exploit")

    def test_timestamp_is_isoformat(self):
        self.assertEqual(_row().to_api_dict()["timestamp"], "2026-05-29T12:00:00+00:00")

    def test_full_observation_preserved(self):
        d = _row().to_api_dict()
        self.assertEqual(d["observation"]["humidity_pct"], 40.0)

    def test_null_fan_defaults_to_zero(self):
        d = _row(action={"fan": None}).to_api_dict()
        self.assertEqual(d["fan_percentage"], 0)

    def test_missing_fan_key_defaults_to_zero(self):
        d = _row(action={}).to_api_dict()
        self.assertEqual(d["fan_percentage"], 0)

    def test_unrewarded_decision(self):
        d = _row(reward=None, reason="safety_override").to_api_dict()
        self.assertIsNone(d["reward"])
        self.assertEqual(d["reason"], "safety_override")


if __name__ == "__main__":
    unittest.main()
