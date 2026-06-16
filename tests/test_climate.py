import json
import os
import unittest
from unittest.mock import patch

from climate import (
    ClimateAction,
    ClimateObservation,
    ClimatePolicy,
    FanAction,
    apply_mist_safety,
    compute_reward,
    discretize,
    safety_override,
    state_key,
)
from qlearning import QLearner

SAFETY_SETTINGS = {
    "CLIMATE_SAFETY_SOC_MIN": 15.0,
    "CLIMATE_SAFETY_TEMP_MAX": 38.0,
    "CLIMATE_SAFETY_HUMIDITY_MAX": 90.0,
}


def _obs(
    air_temp_c=22.0,
    humidity_pct=55.0,
    soc_pct=70.0,
    solar_power_w=150.0,
    forecast_high_c=25.0,
    timestamp="2026-05-29T12:00:00+00:00",
) -> ClimateObservation:
    return ClimateObservation(
        air_temp_c=air_temp_c,
        humidity_pct=humidity_pct,
        soc_pct=soc_pct,
        solar_power_w=solar_power_w,
        forecast_high_c=forecast_high_c,
        timestamp=timestamp,
    )


# ---------------------------------------------------------------------------
# QLearner (generic)
# ---------------------------------------------------------------------------


class TestQLearner(unittest.TestCase):
    def setUp(self):
        self._path = f"/tmp/test_qlearner_{id(self)}.json"
        if os.path.exists(self._path):
            os.remove(self._path)

    def tearDown(self):
        if os.path.exists(self._path):
            os.remove(self._path)

    def test_choose_exploit(self):
        q = QLearner(n_actions=3, epsilon=0.0, model_path=self._path)
        q._q["s1"] = [0.0, 5.0, 1.0]
        idx, explored = q.choose("s1")
        self.assertEqual(idx, 1)
        self.assertFalse(explored)

    def test_choose_explore(self):
        q = QLearner(n_actions=3, epsilon=1.0, model_path=self._path)
        _, explored = q.choose("s1")
        self.assertTrue(explored)

    def test_update_increases_q(self):
        q = QLearner(n_actions=3, epsilon=0.0, model_path=self._path)
        old = q._q["s1"][0]
        q.update("s1", 0, 1.0, "s2")
        self.assertGreater(q._q["s1"][0], old)

    def test_epsilon_decays(self):
        q = QLearner(n_actions=2, epsilon=0.5, epsilon_decay=0.9, model_path=self._path)
        q.update("s1", 0, 0.0, "s2")
        self.assertAlmostEqual(q.epsilon, 0.45)

    def test_epsilon_floors(self):
        q = QLearner(n_actions=2, epsilon=0.06, epsilon_min=0.05, epsilon_decay=0.5, model_path=self._path)
        q.update("s1", 0, 0.0, "s2")
        self.assertAlmostEqual(q.epsilon, 0.05)

    def test_seed(self):
        q = QLearner(n_actions=3, model_path=self._path)
        q.seed("s1", [1.0, 2.0, 3.0])
        self.assertEqual(q._q["s1"], [1.0, 2.0, 3.0])

    def test_save_and_load(self):
        q = QLearner(n_actions=3, epsilon=0.42, model_path=self._path)
        q._q["s1"] = [1.0, 2.0, 3.0]
        q.save()

        loaded = QLearner(n_actions=3, model_path=self._path)
        self.assertEqual(loaded._q["s1"], [1.0, 2.0, 3.0])
        self.assertAlmostEqual(loaded.epsilon, 0.42)

    def test_load_missing_file_no_error(self):
        q = QLearner(n_actions=3, model_path="/tmp/nonexistent_qlearner.json")
        self.assertEqual(len(q._q), 0)

    def test_load_pads_rows_when_action_space_grows(self):
        # Save a 3-action table, reload it as a 5-action learner.
        q = QLearner(n_actions=3, model_path=self._path)
        q._q["s1"] = [1.0, 2.0, 3.0]
        q.save()
        grown = QLearner(n_actions=5, model_path=self._path)
        self.assertEqual(grown._q["s1"], [1.0, 2.0, 3.0, 0.0, 0.0])  # old values kept, new at 0

    def test_load_truncates_rows_when_action_space_shrinks(self):
        q = QLearner(n_actions=5, model_path=self._path)
        q._q["s1"] = [1.0, 2.0, 3.0, 4.0, 5.0]
        q.save()
        shrunk = QLearner(n_actions=3, model_path=self._path)
        self.assertEqual(shrunk._q["s1"], [1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# Discretization
# ---------------------------------------------------------------------------


class TestDiscretize(unittest.TestCase):
    def test_bins_nominal(self):
        obs = _obs(air_temp_c=22.0, humidity_pct=55.0, soc_pct=45.0, solar_power_w=50.0, forecast_high_c=25.0)
        self.assertEqual(discretize(obs), (2, 1, 2, 1, 1))

    def test_bins_extreme_high(self):
        obs = _obs(air_temp_c=40.0, humidity_pct=90.0, soc_pct=80.0, solar_power_w=200.0, forecast_high_c=35.0)
        self.assertEqual(discretize(obs), (4, 3, 3, 2, 2))

    def test_bins_extreme_low(self):
        obs = _obs(air_temp_c=5.0, humidity_pct=20.0, soc_pct=10.0, solar_power_w=0.0, forecast_high_c=10.0)
        self.assertEqual(discretize(obs), (0, 0, 0, 0, 0))

    def test_bin_edges(self):
        # Values exactly on a bin edge land in the upper bin (bisect_right).
        obs = _obs(air_temp_c=13.0, humidity_pct=50.0, soc_pct=15.0, solar_power_w=10.0, forecast_high_c=20.0)
        self.assertEqual(discretize(obs), (1, 1, 1, 1, 1))


# ---------------------------------------------------------------------------
# Safety rails
# ---------------------------------------------------------------------------


class TestSafetyOverride(unittest.TestCase):
    @patch("climate._settings", **SAFETY_SETTINGS)
    def test_low_soc_forces_fan_off(self, _):
        action = safety_override(_obs(soc_pct=10.0))
        self.assertIsNotNone(action)
        self.assertEqual(action.fan.percentage, 0)

    @patch("climate._settings", **SAFETY_SETTINGS)
    def test_extreme_heat_forces_fan_max_and_mist(self, _):
        action = safety_override(_obs(air_temp_c=40.0))
        self.assertIsNotNone(action)
        self.assertEqual(action.fan.percentage, 100)
        self.assertTrue(action.mist)  # emergency cooling enlists the mister

    @patch("climate._settings", **SAFETY_SETTINGS)
    def test_soc_critical_overrides_heat(self, _):
        action = safety_override(_obs(soc_pct=10.0, air_temp_c=40.0))
        self.assertEqual(action.fan.percentage, 0)
        self.assertFalse(action.mist)

    @patch("climate._settings", **SAFETY_SETTINGS)
    def test_nominal_returns_none(self, _):
        self.assertIsNone(safety_override(_obs()))


class TestMistSafety(unittest.TestCase):
    @patch("climate._settings", **SAFETY_SETTINGS)
    def test_high_humidity_forces_mist_off(self, _):
        action = ClimateAction(fan=FanAction(percentage=50), mist=True)
        clamped = apply_mist_safety(_obs(humidity_pct=95.0), action)
        self.assertFalse(clamped.mist)
        self.assertEqual(clamped.fan.percentage, 50)  # fan untouched

    @patch("climate._settings", **SAFETY_SETTINGS)
    def test_mist_allowed_below_ceiling(self, _):
        action = ClimateAction(fan=FanAction(percentage=50), mist=True)
        self.assertTrue(apply_mist_safety(_obs(humidity_pct=70.0), action).mist)

    @patch("climate._settings", **SAFETY_SETTINGS)
    def test_humidity_clamp_overrides_heat_emergency(self, _):
        # 40°C heat emergency wants mist on, but saturated air forces it off.
        action = safety_override(_obs(air_temp_c=40.0, humidity_pct=95.0))
        self.assertFalse(apply_mist_safety(_obs(air_temp_c=40.0, humidity_pct=95.0), action).mist)


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------


class TestComputeReward(unittest.TestCase):
    @staticmethod
    def _configure(mock_settings, comfort_weight=1.0, energy_weight=0.0,
                   humidity_weight=0.0, water_weight=0.0):
        mock_settings.CLIMATE_TEMP_MIN_C = 13.0
        mock_settings.CLIMATE_TEMP_MAX_C = 28.0
        mock_settings.CLIMATE_HUMIDITY_MIN_PCT = 50.0
        mock_settings.CLIMATE_HUMIDITY_MAX_PCT = 70.0
        mock_settings.CLIMATE_REWARD_COMFORT_WEIGHT = comfort_weight
        mock_settings.CLIMATE_REWARD_ENERGY_WEIGHT = energy_weight
        mock_settings.CLIMATE_REWARD_HUMIDITY_WEIGHT = humidity_weight
        mock_settings.CLIMATE_REWARD_WATER_WEIGHT = water_weight
        mock_settings.CLIMATE_SAFETY_SOC_MIN = 15.0
        mock_settings.CLIMATE_SOC_COMFORT = 40.0
        mock_settings.CLIMATE_ENERGY_FLOOR = 0.1
        mock_settings.CLIMATE_ENERGY_FLOOR_NIGHT = 0.5
        mock_settings.CLIMATE_SOLAR_DAYLIGHT_W = 10.0

    @patch("climate._settings")
    def test_in_range_zero_comfort_penalty(self, mock_settings):
        self._configure(mock_settings)
        obs = _obs(air_temp_c=20.0)
        action = ClimateAction(fan=FanAction(percentage=50))
        self.assertEqual(compute_reward(obs, action), 0.0)

    @patch("climate._settings")
    def test_above_range_negative_penalty(self, mock_settings):
        self._configure(mock_settings)
        obs = _obs(air_temp_c=33.0)
        action = ClimateAction(fan=FanAction(percentage=0))
        self.assertAlmostEqual(compute_reward(obs, action), -25.0)

    @patch("climate._settings")
    def test_below_range_negative_penalty(self, mock_settings):
        self._configure(mock_settings)
        obs = _obs(air_temp_c=10.0)
        action = ClimateAction(fan=FanAction(percentage=0))
        self.assertAlmostEqual(compute_reward(obs, action), -9.0)

    @patch("climate._settings")
    def test_energy_cost_full_at_safety_soc(self, mock_settings):
        # At the SOC safety floor, energy is charged at full cost.
        self._configure(mock_settings, comfort_weight=0.0, energy_weight=1.0)
        obs = _obs(air_temp_c=20.0, soc_pct=15.0)
        action = ClimateAction(fan=FanAction(percentage=100))
        self.assertAlmostEqual(compute_reward(obs, action), -1.0)

    @patch("climate._settings")
    def test_energy_scales_between_floor_and_safety(self, mock_settings):
        # Midway (soc=27.5) between safety floor (15) and comfort (40) -> scarcity 0.5.
        self._configure(mock_settings, comfort_weight=0.0, energy_weight=1.0)
        obs = _obs(air_temp_c=20.0, soc_pct=27.5)
        action = ClimateAction(fan=FanAction(percentage=100))
        self.assertAlmostEqual(compute_reward(obs, action), -0.5)

    @patch("climate._settings")
    def test_energy_floor_low_in_daylight(self, mock_settings):
        # Daytime (solar above the cutoff) keeps the low floor -> fan ~free.
        self._configure(mock_settings, comfort_weight=0.0, energy_weight=1.0)
        obs = _obs(air_temp_c=20.0, soc_pct=98.0, solar_power_w=150.0)
        action = ClimateAction(fan=FanAction(percentage=100))
        self.assertAlmostEqual(compute_reward(obs, action), -0.1)

    @patch("climate._settings")
    def test_energy_floor_high_at_night(self, mock_settings):
        # At night (solar ~0) a full battery still pays a real cost (night floor
        # 0.5), so the fan isn't run pointlessly while there is no recharge.
        self._configure(mock_settings, comfort_weight=0.0, energy_weight=1.0)
        obs = _obs(air_temp_c=20.0, soc_pct=98.0, solar_power_w=0.0)
        action = ClimateAction(fan=FanAction(percentage=100))
        self.assertAlmostEqual(compute_reward(obs, action), -0.5)

    @patch("climate._settings")
    def test_low_soc_overrides_daylight_floor(self, mock_settings):
        # SOC scarcity still dominates the floor: a draining battery is expensive
        # even in full sun.
        self._configure(mock_settings, comfort_weight=0.0, energy_weight=1.0)
        obs = _obs(air_temp_c=20.0, soc_pct=15.0, solar_power_w=150.0)
        action = ClimateAction(fan=FanAction(percentage=100))
        self.assertAlmostEqual(compute_reward(obs, action), -1.0)

    @patch("climate._settings")
    def test_no_fan_no_energy_cost(self, mock_settings):
        self._configure(mock_settings, comfort_weight=0.0, energy_weight=1.0)
        obs = _obs(air_temp_c=20.0, soc_pct=15.0, solar_power_w=0.0)
        action = ClimateAction(fan=None)
        self.assertAlmostEqual(compute_reward(obs, action), 0.0)

    @patch("climate._settings")
    def test_humidity_in_band_no_penalty(self, mock_settings):
        self._configure(mock_settings, comfort_weight=0.0, energy_weight=0.0, humidity_weight=1.0)
        obs = _obs(air_temp_c=20.0, humidity_pct=60.0)
        self.assertAlmostEqual(compute_reward(obs, ClimateAction()), 0.0)

    @patch("climate._settings")
    def test_high_humidity_not_penalized(self, mock_settings):
        # Asymmetric: high RH is uncontrollable (can't dehumidify), so no penalty.
        self._configure(mock_settings, comfort_weight=0.0, energy_weight=0.0, humidity_weight=1.0)
        obs = _obs(air_temp_c=20.0, humidity_pct=85.0)
        self.assertAlmostEqual(compute_reward(obs, ClimateAction()), 0.0)

    @patch("climate._settings")
    def test_humidity_too_low_penalty(self, mock_settings):
        self._configure(mock_settings, comfort_weight=0.0, energy_weight=0.0, humidity_weight=1.0)
        obs = _obs(air_temp_c=20.0, humidity_pct=30.0)  # 20 below the 50% floor
        self.assertAlmostEqual(compute_reward(obs, ClimateAction()), -400.0)

    @patch("climate._settings")
    def test_humidity_weighted_below_temperature(self, mock_settings):
        # Same overshoot magnitude (dry side): temperature must dominate humidity.
        self._configure(mock_settings, comfort_weight=1.0, energy_weight=0.0, humidity_weight=0.05)
        obs = _obs(air_temp_c=38.0, humidity_pct=40.0)  # 10°C over, 10% below floor
        # temp: -100 * 1.0 = -100 ; humidity: -100 * 0.05 = -5
        self.assertAlmostEqual(compute_reward(obs, ClimateAction()), -105.0)

    @patch("climate._settings")
    def test_mister_incurs_water_cost(self, mock_settings):
        self._configure(mock_settings, comfort_weight=0.0, energy_weight=0.0, water_weight=0.3)
        obs = _obs(air_temp_c=20.0, humidity_pct=60.0)
        self.assertAlmostEqual(compute_reward(obs, ClimateAction(mist=True)), -0.3)
        self.assertAlmostEqual(compute_reward(obs, ClimateAction(mist=False)), 0.0)


# ---------------------------------------------------------------------------
# ClimatePolicy
# ---------------------------------------------------------------------------


class TestClimatePolicy(unittest.TestCase):
    def setUp(self):
        self._path = f"/tmp/test_climate_policy_{id(self)}.json"
        if os.path.exists(self._path):
            os.remove(self._path)

    def tearDown(self):
        if os.path.exists(self._path):
            os.remove(self._path)

    def _make_policy(self, epsilon=0.0):
        return ClimatePolicy(model_path=self._path, epsilon=epsilon, epsilon_min=0.0)

    def test_action_space_is_fan_times_mist(self):
        policy = self._make_policy()
        self.assertEqual(len(policy.actions), 10)  # 5 fan levels x mist on/off

    def test_decide_exploit_picks_best(self):
        policy = self._make_policy()
        obs = _obs()
        key = state_key(obs)
        # index 3 = (fan 75, mist off) in the mist-off block
        policy._q.seed(key, [0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        action, reason = policy.decide(obs)
        self.assertEqual(action.fan.percentage, 75)
        self.assertFalse(action.mist)
        self.assertEqual(reason, "exploit")

    def test_decide_picks_mist_action(self):
        policy = self._make_policy()
        obs = _obs()
        # index 8 = (fan 75, mist on) in the mist-on block
        policy._q.seed(state_key(obs), [0.0] * 8 + [5.0, 0.0])
        action, _ = policy.decide(obs)
        self.assertEqual(action.fan.percentage, 75)
        self.assertTrue(action.mist)

    def test_action_index_round_trips_mist(self):
        policy = self._make_policy()
        on = ClimateAction(fan=FanAction(percentage=75), mist=True)
        off = ClimateAction(fan=FanAction(percentage=75), mist=False)
        self.assertEqual(policy.actions[policy.action_index(on)], (75, True))
        self.assertEqual(policy.actions[policy.action_index(off)], (75, False))

    def test_decide_explore(self):
        policy = self._make_policy(epsilon=1.0)
        _, reason = policy.decide(_obs())
        self.assertEqual(reason, "explore")

    def test_learn_updates_q(self):
        policy = self._make_policy()
        obs = _obs(air_temp_c=30.0)
        next_obs = _obs(air_temp_c=28.0)
        key = state_key(obs)
        old = policy._q._q[key][2]  # direct access acceptable in tests
        policy.learn(obs, 2, 1.0, next_obs)
        self.assertGreater(policy._q._q[key][2], old)

    def test_action_index_maps_correctly(self):
        policy = self._make_policy()
        self.assertEqual(policy.action_index(ClimateAction(fan=FanAction(percentage=75))), 3)

    def test_action_index_snaps_to_nearest(self):
        policy = self._make_policy()
        self.assertEqual(policy.action_index(ClimateAction(fan=FanAction(percentage=60))), 2)

    def test_warm_start_populates(self):
        policy = self._make_policy()
        self.assertGreater(len(policy._q), 0)


# ---------------------------------------------------------------------------
# Dataclass basics
# ---------------------------------------------------------------------------


class TestDataclasses(unittest.TestCase):
    def test_observation_to_dict(self):
        obs = _obs()
        d = obs.to_dict()
        self.assertEqual(d["air_temp_c"], 22.0)
        self.assertIn("timestamp", d)

    def test_action_to_dict(self):
        action = ClimateAction(fan=FanAction(percentage=50))
        d = action.to_dict()
        self.assertEqual(d["fan"]["percentage"], 50)

    def test_fan_action_clamps(self):
        self.assertEqual(FanAction(percentage=150).percentage, 100)
        self.assertEqual(FanAction(percentage=-10).percentage, 0)

    def test_fan_percentage_property(self):
        self.assertEqual(ClimateAction(fan=FanAction(percentage=75)).fan_percentage, 75)
        self.assertEqual(ClimateAction(fan=None).fan_percentage, 0)

    def test_action_serializable(self):
        action = ClimateAction(fan=FanAction(percentage=75))
        self.assertIsInstance(json.dumps(action.to_dict()), str)


if __name__ == "__main__":
    unittest.main()
