import bisect
import logging
from datetime import datetime, timezone

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
_VOLTAGE_FULL  = 13.8   # snap to 100% — absorption/float complete
_VOLTAGE_EMPTY = 12.0   # snap to 0% — low-voltage cutoff


def _interpolate(table: list, x: float) -> float:
    xs = [p[0] for p in table]
    ys = [p[1] for p in table]
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    i = bisect.bisect_right(xs, x) - 1
    x0, y0 = xs[i], ys[i]
    x1, y1 = xs[i + 1], ys[i + 1]
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


class SolarSOCEstimator:
    def __init__(self, capacity_wh: float):
        self.capacity_wh = capacity_wh
        self.current_wh = capacity_wh * 0.5
        self.last_updated: datetime = datetime.now(timezone.utc)
        self._initialised = False
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
        dt_seconds = (now - self.last_updated).total_seconds()
        self.last_updated = now

        temp_factor = _interpolate(_TEMP_DERATING, battery_temp_c)
        effective_capacity = self.capacity_wh * temp_factor

        # On first run with no saved state, seed from voltage rather than 50%
        if not self._initialised:
            self._initialised = True
            soc = self.voltage_soc(battery_voltage)
            self.current_wh = self.capacity_wh * (soc / 100.0)
            logger.info("Seeded initial SOC from voltage %.2fV: %.1f%%", battery_voltage, soc)
            self._save_state(soc_percent=soc)
            return soc

        # Coulomb counting — clamp to derated capacity so we don't overcount on cold nights
        net_power_w = solar_power_w - load_power_w
        delta_wh = net_power_w * (dt_seconds / 3600.0)
        self.current_wh = max(0.0, min(effective_capacity, self.current_wh + delta_wh))

        # Hard voltage anchors at extremes
        if battery_voltage >= _VOLTAGE_FULL:
            self.current_wh = effective_capacity
        elif battery_voltage <= _VOLTAGE_EMPTY:
            self.current_wh = 0.0

        # Report against nominal capacity so SOC doesn't swing with temperature
        soc = round((self.current_wh / self.capacity_wh) * 100.0, 1)
        self._save_state(soc_percent=soc)
        return soc

    def voltage_soc(self, voltage: float) -> float:
        """Reference-only SOC estimate from resting voltage."""
        return round(_interpolate(_LIFEPO4_VOLTAGE_SOC, voltage), 1)

    @property
    def soc_percent(self) -> float:
        return round((self.current_wh / self.capacity_wh) * 100.0, 1)

    # ------------------------------------------------------------------
    # Persistence via DB
    # ------------------------------------------------------------------

    def _save_state(self, soc_percent: float = None) -> None:
        try:
            from db import get_session
            from models import SolarState
            soc = soc_percent if soc_percent is not None else self.soc_percent
            with get_session() as session:
                state = session.get(SolarState, 1)
                if state is None:
                    session.add(SolarState(id=1, current_wh=self.current_wh, soc_percent=soc, last_updated=self.last_updated))
                else:
                    state.current_wh = self.current_wh
                    state.soc_percent = soc
                    state.last_updated = self.last_updated
        except Exception as e:
            logger.warning("Failed to save solar state: %s", e)

    def _load_state(self) -> None:
        try:
            from db import get_session
            from models import SolarState
            with get_session() as session:
                state = session.get(SolarState, 1)
                if state is not None:
                    self.current_wh = state.current_wh
                    ts = state.last_updated
                    self.last_updated = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
                    self._initialised = True
                    logger.info("Loaded solar state: %.1f Wh (%.1f%%)", self.current_wh, self.soc_percent)
        except Exception as e:
            logger.warning("Failed to load solar state, starting at 50%%: %s", e)
