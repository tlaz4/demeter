import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from solar import SolarSOCEstimator, _interpolate


class TestInterpolate(unittest.TestCase):
    TABLE = [(0.0, 0.0), (10.0, 100.0)]

    def test_below_min_returns_min(self):
        self.assertEqual(_interpolate(self.TABLE, -5.0), 0.0)

    def test_above_max_returns_max(self):
        self.assertEqual(_interpolate(self.TABLE, 20.0), 100.0)

    def test_exact_min(self):
        self.assertEqual(_interpolate(self.TABLE, 0.0), 0.0)

    def test_exact_max(self):
        self.assertEqual(_interpolate(self.TABLE, 10.0), 100.0)

    def test_midpoint(self):
        self.assertEqual(_interpolate(self.TABLE, 5.0), 50.0)

    def test_quarter(self):
        self.assertEqual(_interpolate(self.TABLE, 2.5), 25.0)


class TestVoltageSoc(unittest.TestCase):
    def setUp(self):
        self.est = _make_estimator()

    def test_below_min_returns_zero(self):
        self.assertEqual(self.est.voltage_soc(11.0), 0.0)

    def test_above_max_returns_100(self):
        self.assertEqual(self.est.voltage_soc(14.5), 100.0)

    def test_known_breakpoint_13_6(self):
        self.assertEqual(self.est.voltage_soc(13.6), 98.0)

    def test_known_breakpoint_12_0(self):
        self.assertEqual(self.est.voltage_soc(12.0), 0.0)


class TestCoulombCounting(unittest.TestCase):
    def setUp(self):
        self.est = _make_estimator(current_wh=600.0)

    def test_charging_increases_soc(self):
        soc = _update(self.est, solar_w=100.0, load_w=0.0)
        self.assertGreater(soc, 50.0)

    def test_discharging_decreases_soc(self):
        soc = _update(self.est, solar_w=0.0, load_w=100.0)
        self.assertLess(soc, 50.0)

    def test_balanced_soc_unchanged(self):
        soc = _update(self.est, solar_w=50.0, load_w=50.0)
        self.assertAlmostEqual(soc, 50.0, delta=0.1)

    def test_clamps_at_zero(self):
        self.est.current_wh = 1.0
        soc = _update(self.est, solar_w=0.0, load_w=10000.0)
        self.assertEqual(soc, 0.0)
        self.assertEqual(self.est.current_wh, 0.0)

    def test_clamps_at_capacity(self):
        self.est.current_wh = 1199.0
        soc = _update(self.est, solar_w=10000.0, load_w=0.0)
        self.assertEqual(soc, 100.0)
        self.assertEqual(self.est.current_wh, 1200.0)

    def test_correct_wh_delta(self):
        # 60W net for 60s = 1Wh gained
        self.est.last_updated = datetime.now(timezone.utc) - timedelta(seconds=60)
        _update(self.est, solar_w=60.0, load_w=0.0)
        self.assertAlmostEqual(self.est.current_wh, 601.0, delta=0.05)


class TestVoltageAnchors(unittest.TestCase):
    def setUp(self):
        self.est = _make_estimator(current_wh=800.0)

    def test_upper_anchor_snaps_to_full(self):
        soc = _update(self.est, voltage=13.9)
        self.assertEqual(soc, 100.0)

    def test_lower_anchor_snaps_to_empty(self):
        soc = _update(self.est, voltage=11.9)
        self.assertEqual(soc, 0.0)

    def test_no_anchor_mid_range(self):
        soc = _update(self.est, solar_w=50.0, load_w=50.0, voltage=13.2)
        self.assertGreater(soc, 0.0)
        self.assertLess(soc, 100.0)


class TestTemperatureDerating(unittest.TestCase):
    def setUp(self):
        self.est = _make_estimator(current_wh=600.0)

    def test_cold_clamps_overcharge(self):
        # At 0°C effective capacity = 960Wh — charging beyond that should clamp
        self.est.current_wh = 950.0
        _update(self.est, solar_w=10000.0, load_w=0.0, temp=0.0)
        self.assertLessEqual(self.est.current_wh, 960.0)

    def test_cold_does_not_retroactively_remove_energy(self):
        # Already above the cold ceiling (from a warmer period): a subsequent
        # charge tick in the cold must not snap the accumulator down.
        self.est.current_wh = 1100.0
        _update(self.est, solar_w=10.0, load_w=0.0, temp=0.0)
        self.assertGreaterEqual(self.est.current_wh, 1100.0)

    def test_soc_reported_against_nominal(self):
        # SOC should be stable at 50% regardless of temperature
        soc_warm = _update(self.est, solar_w=50.0, load_w=50.0, temp=25.0)
        soc_cold = _update(self.est, solar_w=50.0, load_w=50.0, temp=0.0)
        self.assertAlmostEqual(soc_warm, soc_cold, delta=0.5)


class TestFirstRunVoltageSeed(unittest.TestCase):
    def test_seeds_from_voltage_on_first_run(self):
        est = _make_estimator(initialised=False)
        with patch.object(est, "_save_state"):
            soc = est.update(solar_power_w=0.0, load_power_w=0.0,
                             battery_voltage=13.6, battery_temp_c=25.0)
        self.assertAlmostEqual(soc, 98.0, delta=0.1)
        self.assertTrue(est._initialised)

    def test_first_run_ignores_coulomb(self):
        # Seed comes from voltage, not from integrating the first tick's net power.
        est = _make_estimator(initialised=False)
        with patch.object(est, "_save_state"):
            soc = est.update(solar_power_w=5.0, load_power_w=5.0,
                             battery_voltage=13.0, battery_temp_c=25.0)
        self.assertAlmostEqual(soc, 10.0, delta=0.5)

    def test_defers_seed_when_under_load(self):
        # Large net discharge means voltage sags below rest — seeding from it
        # would land on the wrong SOC. The estimator should refuse to seed.
        est = _make_estimator(initialised=False)
        soc = _update(est, solar_w=0.0, load_w=200.0, voltage=13.0)
        self.assertFalse(est._initialised)
        # Default unseeded state is 50% (capacity_wh * 0.5)
        self.assertAlmostEqual(soc, 50.0, delta=0.1)

    def test_defers_seed_when_charging_hard(self):
        # Large net charge elevates voltage above rest — also not trustworthy.
        est = _make_estimator(initialised=False)
        soc = _update(est, solar_w=200.0, load_w=0.0, voltage=13.6)
        self.assertFalse(est._initialised)
        self.assertAlmostEqual(soc, 50.0, delta=0.1)

    def test_seeds_on_next_quiet_tick_after_deferral(self):
        est = _make_estimator(initialised=False)
        _update(est, solar_w=0.0, load_w=200.0, voltage=13.0)
        self.assertFalse(est._initialised)
        soc = _update(est, solar_w=5.0, load_w=5.0, voltage=13.6)
        self.assertTrue(est._initialised)
        self.assertAlmostEqual(soc, 98.0, delta=0.1)

    def test_subsequent_run_skips_voltage_seed(self):
        est = _make_estimator(current_wh=600.0, initialised=True)
        soc = _update(est, solar_w=50.0, load_w=50.0, voltage=13.0)
        self.assertAlmostEqual(soc, 50.0, delta=1.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_estimator(current_wh: float = 600.0, initialised: bool = True) -> SolarSOCEstimator:
    with patch.object(SolarSOCEstimator, "_save_state"), \
         patch.object(SolarSOCEstimator, "_load_state"):
        est = SolarSOCEstimator(capacity_wh=1200.0)
        est.current_wh = current_wh
        est._initialised = initialised
        est.last_updated = datetime.now(timezone.utc) - timedelta(seconds=60)
        return est


def _update(est: SolarSOCEstimator, solar_w=0.0, load_w=0.0,
            voltage=13.2, temp=25.0) -> float:
    with patch.object(est, "_save_state"):
        return est.update(solar_power_w=solar_w, load_power_w=load_w,
                          battery_voltage=voltage, battery_temp_c=temp)


if __name__ == "__main__":
    unittest.main()
