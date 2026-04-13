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

    async def push_state(self, entity_id: str, state: str, attributes: dict = None) -> None:
        url = f"{_settings.HA_URL}/api/states/{entity_id}"
        headers = {"Authorization": f"Bearer {_settings.HA_TOKEN}"}
        payload = {"state": state, "attributes": attributes or {}}
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status not in (200, 201):
                    raise HomeAssistantError(f"HA returned {resp.status} pushing state for {entity_id}")
