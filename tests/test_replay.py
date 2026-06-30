import json
import unittest

from rl.climate import ClimateAction, ClimateObservation, ClimatePolicy, FanAction
from rl.replay import (
    Transition,
    build_transitions,
    prepare,
    run_replay,
)


def _obs(temp, ts, soc=70.0, humidity=55.0, solar=150.0, forecast=25.0):
    return ClimateObservation(
        air_temp_c=temp,
        humidity_pct=humidity,
        soc_pct=soc,
        solar_power_w=solar,
        forecast_high_c=forecast,
        timestamp=ts,
    )


def _row(obs: ClimateObservation, action: ClimateAction) -> tuple[str, str]:
    """Mirror how DecisionLog stores a step: raw obs/action JSON."""
    return json.dumps(obs.to_dict()), json.dumps(action.to_dict())


def _act(fan, mist=False):
    return ClimateAction(fan=FanAction(percentage=fan), mist=mist)


T0 = "2026-06-01T12:00:00+00:00"
T2 = "2026-06-01T12:02:00+00:00"  # +120s (one interval)
T4 = "2026-06-01T12:04:00+00:00"
T_FAR = "2026-06-01T18:00:00+00:00"  # big gap (workflow down)


class BuildTransitionsTests(unittest.TestCase):
    def test_pairs_consecutive_rows(self):
        rows = [
            _row(_obs(30, T0), _act(100)),
            _row(_obs(26, T2), _act(0)),
            _row(_obs(25, T4), _act(0)),
        ]
        trans = build_transitions(rows, max_gap_s=240)
        self.assertEqual(len(trans), 2)
        self.assertEqual(trans[0].obs.air_temp_c, 30)
        self.assertEqual(trans[0].next_obs.air_temp_c, 26)

    def test_drops_pair_across_large_time_gap(self):
        rows = [
            _row(_obs(30, T0), _act(100)),
            _row(_obs(26, T_FAR), _act(0)),  # gap far exceeds max
            _row(_obs(25, T_FAR.replace("18:00", "18:02")), _act(0)),
        ]
        trans = build_transitions(rows, max_gap_s=240)
        # Only the last pair (T_FAR -> T_FAR+2m) is within the gap window.
        self.assertEqual(len(trans), 1)
        self.assertEqual(trans[0].obs.air_temp_c, 26)

    def test_unparseable_row_does_not_pair_across(self):
        rows = [
            _row(_obs(30, T0), _act(100)),
            ("{bad json", "{bad}"),
            _row(_obs(25, T4), _act(0)),
        ]
        trans = build_transitions(rows, max_gap_s=10_000)
        self.assertEqual(trans, [])


class PrepareTests(unittest.TestCase):
    def setUp(self):
        self.policy = ClimatePolicy(model_path="__test_no_load__")

    def test_drops_action_not_in_current_space(self):
        # fan=50 is no longer a valid level ([0, 85, 100]).
        trans = [Transition(_obs(30, T0), _act(50), _obs(26, T2))]
        samples, dropped = prepare(self.policy, trans, snap=False)
        self.assertEqual(samples, [])
        self.assertEqual(dropped, 1)

    def test_snap_keeps_removed_action(self):
        trans = [Transition(_obs(30, T0), _act(50), _obs(26, T2))]
        samples, dropped = prepare(self.policy, trans, snap=True)
        self.assertEqual(len(samples), 1)
        self.assertEqual(dropped, 0)

    def test_valid_action_resolves_to_index(self):
        trans = [Transition(_obs(30, T0), _act(100, mist=False), _obs(26, T2))]
        samples, _ = prepare(self.policy, trans, snap=False)
        state_key, idx, reward, next_key = samples[0]
        self.assertEqual(self.policy.actions[idx], (100, False))
        self.assertIsInstance(reward, float)


class ReplayLearningTests(unittest.TestCase):
    def test_replay_prefers_action_with_higher_reward(self):
        policy = ClimatePolicy(model_path="__test_no_load__")
        hot = _obs(32, T0)  # same start state for both transitions

        # Action A: strong cooling -> next obs back in range (good reward).
        # Action B: do nothing -> stays hot (large comfort penalty).
        good = Transition(hot, _act(100), _obs(26, T2))
        bad = Transition(hot, _act(0), _obs(33, T2))
        samples, dropped = prepare(policy, [good, bad], snap=False)
        self.assertEqual(dropped, 0)

        eps_before = policy._q.epsilon
        run_replay(policy, samples, epochs=200)

        s = list({s for s, _, _, _ in samples})[0]
        q = policy._q._q[s]
        idx_cool = policy.actions.index((100, False))
        idx_off = policy.actions.index((0, False))
        self.assertGreater(q[idx_cool], q[idx_off])
        # Epsilon restored despite update()'s per-call decay.
        self.assertAlmostEqual(policy._q.epsilon, eps_before)


if __name__ == "__main__":
    unittest.main()
