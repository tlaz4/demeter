import unittest
from unittest.mock import AsyncMock, patch

from home_assistant import HomeAssistantClient, HomeAssistantError


def make_state(state: str, attributes: dict = None) -> dict:
    return {"state": state, "attributes": attributes or {}}


class TestGetLoadPowerW(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.client = HomeAssistantClient()

    async def _get_load(self, loads: list, states: list) -> float:
        self.client.get_state = AsyncMock(side_effect=states)
        with patch("home_assistant._settings") as s:
            s.LOADS = loads
            return await self.client.get_load_power_w()

    async def test_binary_on(self):
        result = await self._get_load(
            [{"name": "light", "entity_id": "switch.light", "power_w": 60.0, "type": "binary"}],
            [make_state("on")],
        )
        self.assertEqual(result, 60.0)

    async def test_binary_off(self):
        result = await self._get_load(
            [{"name": "light", "entity_id": "switch.light", "power_w": 60.0, "type": "binary"}],
            [make_state("off")],
        )
        self.assertEqual(result, 0.0)

    async def test_binary_unavailable(self):
        result = await self._get_load(
            [{"name": "light", "entity_id": "switch.light", "power_w": 60.0, "type": "binary"}],
            [make_state("unavailable")],
        )
        self.assertEqual(result, 0.0)

    async def test_percentage_at_50(self):
        result = await self._get_load(
            [{"name": "fan", "entity_id": "fan.fan", "power_w": 24.0, "type": "percentage"}],
            [make_state("on", {"percentage": 50})],
        )
        self.assertAlmostEqual(result, 12.0)

    async def test_percentage_off(self):
        result = await self._get_load(
            [{"name": "fan", "entity_id": "fan.fan", "power_w": 24.0, "type": "percentage"}],
            [make_state("off", {"percentage": 50})],
        )
        self.assertEqual(result, 0.0)

    async def test_percentage_full(self):
        result = await self._get_load(
            [{"name": "fan", "entity_id": "fan.fan", "power_w": 24.0, "type": "percentage"}],
            [make_state("on", {"percentage": 100})],
        )
        self.assertAlmostEqual(result, 24.0)

    async def test_sensor_type(self):
        result = await self._get_load(
            [{"name": "mppt", "entity_id": "sensor.mppt", "power_w": 0.0, "type": "sensor"}],
            [make_state("18.5")],
        )
        self.assertAlmostEqual(result, 18.5)

    async def test_sensor_unavailable(self):
        result = await self._get_load(
            [{"name": "mppt", "entity_id": "sensor.mppt", "power_w": 0.0, "type": "sensor"}],
            [make_state("unavailable")],
        )
        self.assertEqual(result, 0.0)

    async def test_ha_error_assumes_off(self):
        result = await self._get_load(
            [{"name": "light", "entity_id": "switch.light", "power_w": 60.0, "type": "binary"}],
            [HomeAssistantError("timeout")],
        )
        self.assertEqual(result, 0.0)

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
