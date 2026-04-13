import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from home_assistant import HomeAssistantClient, HomeAssistantError


class TestGetState(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.client = HomeAssistantClient()

    async def test_returns_state_dict(self):
        mock_response = {"state": "on", "attributes": {}}
        with patch("home_assistant.aiohttp.ClientSession") as mock_session:
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.json = AsyncMock(return_value=mock_response)
            mock_get = MagicMock()
            mock_get.__aenter__ = AsyncMock(return_value=mock_resp)
            mock_get.__aexit__ = AsyncMock(return_value=None)
            mock_session.return_value.__aenter__.return_value.get = MagicMock(return_value=mock_get)
            result = await self.client.get_state("switch.light")
        self.assertEqual(result, mock_response)

    async def test_raises_on_non_200(self):
        with patch("home_assistant.aiohttp.ClientSession") as mock_session:
            mock_resp = AsyncMock()
            mock_resp.status = 404
            mock_get = MagicMock()
            mock_get.__aenter__ = AsyncMock(return_value=mock_resp)
            mock_get.__aexit__ = AsyncMock(return_value=None)
            mock_session.return_value.__aenter__.return_value.get = MagicMock(return_value=mock_get)
            with self.assertRaises(HomeAssistantError):
                await self.client.get_state("switch.nonexistent")


if __name__ == "__main__":
    unittest.main()
