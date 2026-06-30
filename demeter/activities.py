import asyncio
import logging
import time
from datetime import datetime, timezone

from temporalio import activity

logger = logging.getLogger(__name__)

import json

import settings as _settings
from rl.climate import (
    ClimateAction,
    ClimateObservation,
    ClimatePolicy,
    apply_mist_safety,
    compute_reward,
    safety_override,
)
from db import get_session, init_db
from home_assistant import HomeAssistantClient, HomeAssistantError
from models import DecisionLog
from solar import SolarHAClient, SolarSOCEstimator


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
        self._solar_ha = SolarHAClient(HomeAssistantClient())

    @activity.defn
    async def poll_solar(self) -> dict:
        logger.info("Polling solar data from Home Assistant")
        solar_data = await self._solar_ha.get_solar_data()
        load_power_w = await self._solar_ha.get_load_power_w()

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
            await self._solar_ha.push_soc(soc)
            logger.info("Pushed SOC to HA: %s = %.1f%%", _settings.HA_ENTITY_SOC, soc)
        except HomeAssistantError as e:
            logger.warning("Failed to push SOC to HA: %s", e)

        return {
            "soc_percent": soc,
            "energy_wh": round(self._estimator.current_wh, 1),
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }


class ClimateControlActivities:
    def __init__(self):
        init_db()
        self._ha = HomeAssistantClient()
        self._policy = ClimatePolicy()
        self._prev_obs: ClimateObservation | None = None
        self._prev_action: ClimateAction | None = None
        self._prev_log_id: int | None = None

    @activity.defn
    async def run_climate_control(self) -> dict:
        obs = await self._observe()

        # TD update: reward the previous action now that we can see its effect
        if self._prev_obs is not None and self._prev_action is not None:
            reward = compute_reward(obs, self._prev_action)
            action_idx = self._policy.action_index(self._prev_action)
            self._policy.learn(self._prev_obs, action_idx, reward, obs)
            self._update_log_reward(self._prev_log_id, reward)
        else:
            reward = None

        # Safety rails first, then Q-learning
        override = safety_override(obs)
        if override is not None:
            action = override
            reason = "safety_override"
            policy_name = "safety"
        else:
            action, reason = self._policy.decide(obs)
            policy_name = self._policy.name

        # Hard humidity rail on the mister, applied to whatever was chosen.
        action = apply_mist_safety(obs, action)

        await self._execute(action)
        self._prev_log_id = self._log_decision(obs, action, policy_name, reason)

        logger.info(
            "Climate: %.1f°C / %.1f%% RH | SOC %.1f%% | Forecast %.0f°C | Fan %d%% | Mist %s | %s (%s) | reward=%s",
            obs.air_temp_c, obs.humidity_pct, obs.soc_pct, obs.forecast_high_c,
            action.fan_percentage, "on" if action.mist else "off", reason, policy_name,
            f"{reward:.3f}" if reward is not None else "n/a",
        )

        self._prev_obs = obs
        self._prev_action = action

        return {
            "observation": obs.to_dict(),
            "action": action.to_dict(),
            "policy": policy_name,
            "reason": reason,
        }

    async def _observe(self) -> ClimateObservation:
        all_entities = list(_settings.HA_ENTITY_AIR_TEMPS) + [
            _settings.HA_ENTITY_HUMIDITY,
            _settings.HA_ENTITY_SOC,
            _settings.HA_ENTITY_SOLAR_POWER,
            _settings.HA_ENTITY_WEATHER_FORECAST,
        ]
        results = await asyncio.gather(
            *(self._ha.get_state(e) for e in all_entities),
            return_exceptions=True,
        )

        n_temps = len(_settings.HA_ENTITY_AIR_TEMPS)
        temp_readings: dict[str, float] = {}
        for entity_id, result in zip(_settings.HA_ENTITY_AIR_TEMPS, results[:n_temps]):
            if isinstance(result, Exception):
                logger.warning("Failed to read %s: %s", entity_id, result)
            else:
                temp_readings[entity_id] = float(result["state"])

        avg_temp = sum(temp_readings.values()) / len(temp_readings) if temp_readings else 25.0

        hum, soc, solar, forecast = results[n_temps:]
        return ClimateObservation(
            air_temp_c=avg_temp,
            humidity_pct=float(hum["state"]),
            soc_pct=float(soc["state"]),
            solar_power_w=float(solar["state"]),
            forecast_high_c=float(forecast["state"]),
            timestamp=datetime.now(timezone.utc).isoformat(),
            temp_readings=temp_readings,
        )

    async def _execute(self, action: ClimateAction) -> None:
        if action.fan is not None:
            entity_id = _settings.HA_ENTITY_FAN
            if action.fan.percentage == 0:
                await self._ha.call_service("fan", "turn_off", {"entity_id": entity_id})
            else:
                await self._ha.call_service("fan", "set_percentage", {
                    "entity_id": entity_id,
                    "percentage": action.fan.percentage,
                })

        # Mister is an on/off switch entity in HA.
        mister = _settings.HA_ENTITY_MISTER
        service = "turn_on" if action.mist else "turn_off"
        await self._ha.call_service("switch", service, {"entity_id": mister})

    def _log_decision(
        self,
        obs: ClimateObservation,
        action: ClimateAction,
        policy_name: str,
        reason: str,
    ) -> int | None:
        try:
            with get_session() as session:
                row = DecisionLog(
                    timestamp=datetime.now(timezone.utc),
                    observation_json=json.dumps(obs.to_dict()),
                    action_json=json.dumps(action.to_dict()),
                    policy_name=policy_name,
                    reason=reason,
                    reward=None,
                )
                session.add(row)
                session.flush()
                return row.id
        except Exception as e:
            logger.warning("Failed to log climate decision: %s", e)
            return None

    def _update_log_reward(self, log_id: int | None, reward: float) -> None:
        if log_id is None:
            return
        try:
            with get_session() as session:
                row = session.get(DecisionLog, log_id)
                if row:
                    row.reward = reward
        except Exception as e:
            logger.warning("Failed to update reward for log %s: %s", log_id, e)
