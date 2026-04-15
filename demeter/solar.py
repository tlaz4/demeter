import bisect
import logging
from datetime import datetime, timezone

import settings as _settings
from db import get_session
from models import SolarState
from home_assistant import HomeAssistantError

logger = logging.getLogger(__name__)

# LiFePO4 4S (12V pack) resting voltage → SOC lookup table
# Voltage is highly compressed in the middle — only reliable at extremes
_LIFEPO4_VOLTAGE_SOC = [
    (12.0,  0.0),
    (12.5,  2.0),
    (12.8,  5.0),
    (13.0, 10.0),
    (13.1, 20.0),
    (13.15, 30.0),
    (13.2, 40.0),
    (13.25, 60.0),
    (13.3, 80.0),
    (13.4, 95.0),
    (13.6, 98.0),
    (13.8, 100.0),
]

# Temperature derating: capacity factor at each temperature (°C)
_TEMP_DERATING = [
    (-20, 0.40),
    (-10, 0.60),
    (  0, 0.80),
    ( 10, 0.92),
    ( 20, 0.98),
    ( 25, 1.00),
    ( 40, 1.00),
]

# Voltage thresholds for hard anchoring
_VOLTAGE_FULL          = 13.8   # snap to 100% — absorption/float complete
_VOLTAGE_FULL_RELEASE  = 13.5   # must drop below this before leaving anchored-full state
_VOLTAGE_EMPTY         = 12.0   # snap to 0% — low-voltage cutoff
_VOLTAGE_EMPTY_RELEASE = 12.3   # must rise above this before leaving anchored-empty state

# Cap integration window to guard against long offline gaps producing huge deltas
_MAX_DT_SECONDS = 300.0

# Net power threshold below which the battery is considered quiescent enough for
# voltage-based SOC seeding to be trustworthy. Above this, terminal voltage is
# distorted by IR drop/boost and the lookup table cannot be trusted.
_SEED_QUIESCENCE_W = 20.0


def _interpolate(table: list, x: float) -> float:
    if x <= table[0][0]:
        return table[0][1]
    if x >= table[-1][0]:
        return table[-1][1]
    i = bisect.bisect_right([p[0] for p in table], x) - 1
    (x0, y0), (x1, y1) = table[i], table[i + 1]
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def _pct(wh: float, capacity_wh: float) -> float:
    return round((wh / capacity_wh) * 100.0, 1)


class SolarSOCEstimator:
    def __init__(self, capacity_wh: float):
        if capacity_wh <= 0:
            raise ValueError(f"capacity_wh must be positive, got {capacity_wh}")
        self.capacity_wh = capacity_wh
        self.current_wh = capacity_wh * 0.5
        self.last_updated: datetime = datetime.now(timezone.utc)
        self._initialised = False
        self._anchored_full = False
        self._anchored_empty = False
        self._load_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        solar_power_w: float,
        load_power_w: float,
        battery_voltage: float,
        battery_temp_c: float,
    ) -> float:
        now = datetime.now(timezone.utc)
        dt_seconds = min((now - self.last_updated).total_seconds(), _MAX_DT_SECONDS)
        self.last_updated = now

        if not self._initialised:
            net_power_w = solar_power_w - load_power_w
            if abs(net_power_w) > _SEED_QUIESCENCE_W:
                logger.info(
                    "Deferring voltage seed: net power %.1fW exceeds quiescence threshold %.1fW",
                    net_power_w, _SEED_QUIESCENCE_W,
                )
                return self.soc_percent
            return self._seed_from_voltage(battery_voltage)

        self._integrate(solar_power_w, load_power_w, dt_seconds, battery_temp_c)
        soc = self._apply_voltage_anchors(battery_voltage)
        self._save_state()
        return soc

    @staticmethod
    def voltage_soc(voltage: float) -> float:
        """Reference-only SOC estimate from resting voltage."""
        return round(_interpolate(_LIFEPO4_VOLTAGE_SOC, voltage), 1)

    @property
    def soc_percent(self) -> float:
        return _pct(self.current_wh, self.capacity_wh)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _seed_from_voltage(self, battery_voltage: float) -> float:
        self._initialised = True
        soc = self.voltage_soc(battery_voltage)
        self.current_wh = self.capacity_wh * (soc / 100.0)
        logger.info("Seeded initial SOC from voltage %.2fV: %.1f%%", battery_voltage, soc)
        self._save_state()
        return soc

    def _integrate(
        self,
        solar_power_w: float,
        load_power_w: float,
        dt_seconds: float,
        battery_temp_c: float,
    ) -> None:
        # Coulomb counting — when charging, cap new energy at the derated ceiling but
        # never trim energy already counted, so a temperature dip doesn't retroactively
        # remove energy from the accumulator.
        net_power_w = solar_power_w - load_power_w
        delta_wh = net_power_w * (dt_seconds / 3600.0)
        new_wh = max(0.0, self.current_wh + delta_wh)
        if net_power_w > 0:
            effective_capacity = self.capacity_wh * _interpolate(_TEMP_DERATING, battery_temp_c)
            ceiling = max(effective_capacity, self.current_wh)
            new_wh = min(new_wh, ceiling)
        self.current_wh = new_wh

    def _apply_voltage_anchors(self, battery_voltage: float) -> float:
        # Hysteresis on both ends: latch when voltage crosses the anchor, only
        # release once it's moved clearly past the release threshold.
        if battery_voltage >= _VOLTAGE_FULL:
            self._anchored_full = True
        elif battery_voltage < _VOLTAGE_FULL_RELEASE:
            self._anchored_full = False

        if battery_voltage <= _VOLTAGE_EMPTY:
            self._anchored_empty = True
        elif battery_voltage > _VOLTAGE_EMPTY_RELEASE:
            self._anchored_empty = False

        if self._anchored_full:
            self.current_wh = self.capacity_wh
        elif self._anchored_empty:
            self.current_wh = 0.0
        return self.soc_percent

    # ------------------------------------------------------------------
    # Persistence via DB
    # ------------------------------------------------------------------

    def _save_state(self) -> None:
        try:
            with get_session() as session:
                state = session.get(SolarState, 1)
                if state is None:
                    session.add(SolarState(
                        id=1,
                        current_wh=self.current_wh,
                        soc_percent=self.soc_percent,
                        last_updated=self.last_updated,
                    ))
                else:
                    state.current_wh = self.current_wh
                    state.soc_percent = self.soc_percent
                    state.last_updated = self.last_updated
        except Exception as e:
            logger.warning("Failed to save solar state: %s", e)

    def _load_state(self) -> None:
        try:
            with get_session() as session:
                state = session.get(SolarState, 1)
                if state is None:
                    return
                self.current_wh = state.current_wh
                ts = state.last_updated
                self.last_updated = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
                self._initialised = True
                # Anchors must be re-established from live voltage, not stale SOC —
                # otherwise a saved 100% locks us to full until voltage drops below
                # the release threshold.
                logger.info("Loaded solar state: %.1f Wh (%.1f%%)", self.current_wh, self.soc_percent)
        except Exception as e:
            logger.warning("Failed to load solar state, starting at 50%%: %s", e)


class SolarHAClient:
    """Solar-specific Home Assistant interactions."""

    def __init__(self, ha_client):
        self._ha = ha_client

    async def get_solar_data(self) -> dict:
        voltage_state = await self._ha.get_state(_settings.HA_ENTITY_BATTERY_VOLTAGE)
        power_state   = await self._ha.get_state(_settings.HA_ENTITY_SOLAR_POWER)
        temp_state    = await self._ha.get_state(_settings.HA_ENTITY_BATTERY_TEMP)

        try:
            return {
                "battery_voltage": float(voltage_state["state"]),
                "solar_power_w":   float(power_state["state"]),
                "battery_temp_c":  float(temp_state["state"]),
            }
        except (KeyError, ValueError) as e:
            raise HomeAssistantError(f"Failed to parse solar data: {e}") from e

    async def get_load_power_w(self) -> float:
        loads = _settings.LOADS
        total = 0.0
        failures = 0
        for load in loads:
            try:
                state = await self._ha.get_state(load.get("entity_id"))
                total += self._calc_load_power(state, load)
            except HomeAssistantError:
                failures += 1
                logger.warning("Could not get state for load '%s', assuming off", load.get("name"))
        if loads and failures == len(loads):
            # All reads failed — treating as zero load would silently inflate SOC.
            raise HomeAssistantError(f"All {failures} load state reads failed")
        return total

    def _calc_load_power(self, state: dict, load: dict) -> float:
        load_type = load.get("type")
        handler = self._LOAD_HANDLERS.get(load_type)
        if handler is None:
            logger.warning("Unknown load type '%s' for load '%s'", load_type, load.get("name"))
            return 0.0
        return handler(self, state, load)

    def _load_binary(self, state: dict, load: dict) -> float:
        raw = state.get("state", "unavailable")
        if raw not in ("on", "off", "unavailable"):
            logger.warning("Unexpected state '%s' for load '%s'", raw, load.get("name"))
        return load.get("power_w", 0.0) if raw == "on" else 0.0

    def _load_percentage(self, state: dict, load: dict) -> float:
        raw = state.get("state", "unavailable")
        if raw in ("off", "unavailable", "unknown"):
            return 0.0
        pct = float(state.get("attributes", {}).get("percentage") or 0) / 100.0
        return load.get("power_w", 0.0) * pct

    def _load_sensor(self, state: dict, _load: dict) -> float:
        raw = state.get("state", "unavailable")
        try:
            return float(raw)
        except (TypeError, ValueError):
            return 0.0

    _LOAD_HANDLERS: dict = {
        "binary":     _load_binary,
        "percentage": _load_percentage,
        "sensor":     _load_sensor,
    }

    async def push_soc(self, soc: float) -> None:
        await self._ha.push_state(
            _settings.HA_ENTITY_SOC,
            state=f"{soc:.1f}",
            attributes={
                "unit_of_measurement": "%",
                "friendly_name": "Battery SOC",
                "device_class": "battery",
            },
        )
