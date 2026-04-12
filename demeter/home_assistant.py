import logging

import aiohttp

try:
    from demeter import settings as _settings
except ImportError:
    import settings as _settings

logger = logging.getLogger(__name__)


class HomeAssistantError(Exception):
    pass


class HomeAssistantClient:
    async def get_state(self, entity_id: str) -> dict:
        url = f"{_settings.HA_URL}/api/states/{entity_id}"
        headers = {"Authorization": f"Bearer {_settings.HA_TOKEN}"}
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    raise HomeAssistantError(f"HA returned {resp.status} for {entity_id}")
                return await resp.json()

    async def get_solar_data(self) -> dict:
        voltage_state = await self.get_state(_settings.HA_ENTITY_BATTERY_VOLTAGE)
        power_state   = await self.get_state(_settings.HA_ENTITY_SOLAR_POWER)
        temp_state    = await self.get_state(_settings.HA_ENTITY_BATTERY_TEMP)

        try:
            return {
                "battery_voltage": float(voltage_state["state"]),
                "solar_power_w":   float(power_state["state"]),
                "battery_temp_c":  float(temp_state["state"]),
            }
        except (KeyError, ValueError) as e:
            raise HomeAssistantError(f"Failed to parse solar data: {e}") from e

    async def push_state(self, entity_id: str, state: str, attributes: dict = None) -> None:
        url = f"{_settings.HA_URL}/api/states/{entity_id}"
        headers = {"Authorization": f"Bearer {_settings.HA_TOKEN}"}
        payload = {"state": state, "attributes": attributes or {}}
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status not in (200, 201):
                    raise HomeAssistantError(f"HA returned {resp.status} pushing state for {entity_id}")

    async def get_load_power_w(self) -> float:
        total = 0.0
        for load in _settings.LOADS:
            try:
                state = await self.get_state(load["entity_id"])
                raw = state["state"]

                if load["type"] == "binary":
                    if raw not in ("on", "off", "unavailable"):
                        logger.warning("Unexpected state '%s' for load '%s'", raw, load["name"])
                    total += load["power_w"] if raw == "on" else 0.0

                elif load["type"] == "percentage":
                    if raw == "off":
                        pct = 0.0
                    else:
                        pct = float(state["attributes"].get("percentage") or 0) / 100.0
                    total += load["power_w"] * pct

                elif load["type"] == "sensor":
                    total += float(raw) if raw not in ("unavailable", "unknown") else 0.0

            except HomeAssistantError:
                logger.warning("Could not get state for load '%s', assuming off", load["name"])

        return total
