import unittest
from unittest.mock import AsyncMock, patch

from home_assistant import HomeAssistantError
from solar import SolarHAClient


def make_state(state: str, attributes: dict = None) -> dict:
    return {"state": state, "attributes": attributes or {}}


def make_load(load_type: str, name: str = "load", power_w: float = 0.0) -> dict:
    return {"type": load_type, "name": name, "power_w": power_w, "entity_id": f"x.{name}"}


def make_client(loads: list) -> SolarHAClient:
    ha = AsyncMock()
    with patch("solar._settings") as s:
        s.LOADS = loads
    client = SolarHAClient(ha)
    return client, ha


class TestCalcLoadPower(unittest.TestCase):
    """Tests for _calc_load_power — synchronous, no HA calls needed."""

    def setUp(self):
        self.client = SolarHAClient(AsyncMock())

    def test_binary_on(self):
        result = self.client._calc_load_power(make_state("on"), make_load("binary", "light", 60.0))
        self.assertEqual(result, 60.0)

    def test_binary_off(self):
        result = self.client._calc_load_power(make_state("off"), make_load("binary", "light", 60.0))
        self.assertEqual(result, 0.0)

    def test_binary_unavailable(self):
        result = self.client._calc_load_power(make_state("unavailable"), make_load("binary", "light", 60.0))
        self.assertEqual(result, 0.0)

    def test_percentage_at_50(self):
        result = self.client._calc_load_power(make_state("on", {"percentage": 50}), make_load("percentage", "fan", 24.0))
        self.assertAlmostEqual(result, 12.0)

    def test_percentage_off(self):
        result = self.client._calc_load_power(make_state("off", {"percentage": 50}), make_load("percentage", "fan", 24.0))
        self.assertEqual(result, 0.0)

    def test_percentage_full(self):
        result = self.client._calc_load_power(make_state("on", {"percentage": 100}), make_load("percentage", "fan", 24.0))
        self.assertAlmostEqual(result, 24.0)

    def test_sensor_type(self):
        result = self.client._calc_load_power(make_state("18.5"), make_load("sensor", "mppt"))
        self.assertAlmostEqual(result, 18.5)

    def test_sensor_unavailable(self):
        result = self.client._calc_load_power(make_state("unavailable"), make_load("sensor", "mppt"))
        self.assertEqual(result, 0.0)

    def test_sensor_unknown(self):
        result = self.client._calc_load_power(make_state("unknown"), make_load("sensor", "mppt"))
        self.assertEqual(result, 0.0)

    def test_unknown_type_returns_zero(self):
        result = self.client._calc_load_power(make_state("on"), make_load("unsupported", "thing", 100.0))
        self.assertEqual(result, 0.0)


class TestGetLoadPowerW(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.ha = AsyncMock()
        self.client = SolarHAClient(self.ha)

    async def _get_load(self, loads, states):
        self.ha.get_state = AsyncMock(side_effect=states)
        with patch("solar._settings") as s:
            s.LOADS = loads
            return await self.client.get_load_power_w()

    async def test_partial_ha_error_assumes_off(self):
        # One load fails, another succeeds — the failed one is treated as off
        # and the total reflects only the successful read.
        result = await self._get_load(
            [
                {"name": "light", "entity_id": "switch.light", "power_w": 60.0, "type": "binary"},
                {"name": "fan",   "entity_id": "fan.fan",       "power_w": 24.0, "type": "binary"},
            ],
            [HomeAssistantError("timeout"), make_state("on")],
        )
        self.assertEqual(result, 24.0)

    async def test_all_ha_errors_raises(self):
        # If every load read fails, returning 0 would silently inflate SOC;
        # caller must see the failure and retry.
        with self.assertRaises(HomeAssistantError):
            await self._get_load(
                [{"name": "light", "entity_id": "switch.light", "power_w": 60.0, "type": "binary"}],
                [HomeAssistantError("timeout")],
            )

    async def test_multiple_loads_summed(self):
        result = await self._get_load(
            [
                {"name": "light", "entity_id": "switch.light", "power_w": 60.0, "type": "binary"},
                {"name": "fan",   "entity_id": "fan.fan",       "power_w": 24.0, "type": "percentage"},
                {"name": "mppt",  "entity_id": "sensor.mppt",   "power_w": 0.0,  "type": "sensor"},
            ],
            [
                make_state("on"),
                make_state("on", {"percentage": 50}),
                make_state("10.0"),
            ],
        )
        self.assertAlmostEqual(result, 82.0)


if __name__ == "__main__":
    unittest.main()
