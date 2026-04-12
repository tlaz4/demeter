import asyncio
import logging
import time
from datetime import datetime, timezone

from temporalio import activity

logger = logging.getLogger(__name__)

import settings as _settings
from db import init_db
from home_assistant import HomeAssistantClient, HomeAssistantError
from solar import SolarSOCEstimator


def do_something(obj_id):
    time.sleep(1)
    return "Snap!"


class PlantSnapshotActvities:
    @activity.defn
    async def take_snapshot(self, obj_id) -> str:
        snapshot_result = await asyncio.to_thread(
            do_something, obj_id
        )
        return snapshot_result


class SolarPollActivities:
    def __init__(self):
        init_db()
        self._estimator = SolarSOCEstimator(capacity_wh=_settings.BATTERY_CAPACITY_WH)
        self._ha = HomeAssistantClient()

    @activity.defn
    async def poll_solar(self) -> dict:
        logger.info("Polling solar data from Home Assistant")
        solar_data = await self._ha.get_solar_data()
        load_power_w = await self._ha.get_load_power_w()

        logger.info(
            "Solar: %.1fW in | Load: %.1fW out | Battery: %.2fV | Temp: %.1f°C",
            solar_data["solar_power_w"],
            load_power_w,
            solar_data["battery_voltage"],
            solar_data["battery_temp_c"],
        )

        soc = self._estimator.update(
            solar_power_w=solar_data["solar_power_w"],
            load_power_w=load_power_w,
            battery_voltage=solar_data["battery_voltage"],
            battery_temp_c=solar_data["battery_temp_c"],
        )

        logger.info("SOC estimate: %.1f%% (%.1f Wh)", soc, self._estimator.current_wh)

        try:
            await self._ha.push_state(
                _settings.HA_ENTITY_SOC,
                state=str(soc),
                attributes={"unit_of_measurement": "%", "friendly_name": "Battery SOC", "device_class": "battery"},
            )
            logger.info("Pushed SOC to HA: %s = %.1f%%", _settings.HA_ENTITY_SOC, soc)
        except HomeAssistantError as e:
            logger.warning("Failed to push SOC to HA: %s", e)

        return {
            "soc_percent": soc,
            "energy_wh": round(self._estimator.current_wh, 1),
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
