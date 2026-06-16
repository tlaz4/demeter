import itertools
import logging
from bisect import bisect_right
from dataclasses import asdict, dataclass, field

try:
    from demeter import settings as _settings
except ImportError:
    import settings as _settings

from qlearning import QLearner

logger = logging.getLogger(__name__)

FAN_LEVELS = [0, 25, 50, 75, 100]
MIST_LEVELS = [False, True]


@dataclass(frozen=True)
class ClimateObservation:
    air_temp_c: float
    humidity_pct: float
    soc_pct: float
    solar_power_w: float
    forecast_high_c: float
    timestamp: str
    temp_readings: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class FanAction:
    percentage: int

    def __post_init__(self):
        object.__setattr__(self, "percentage", max(0, min(100, self.percentage)))


@dataclass(frozen=True)
class ClimateAction:
    fan: FanAction | None = None
    mist: bool = False

    @property
    def fan_percentage(self) -> int:
        return self.fan.percentage if self.fan else 0

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# State discretization
# ---------------------------------------------------------------------------

BIN_EDGES = {
    "temp": [13, 20, 28, 35],
    # Edges aligned to the humidity comfort band (50/70) with the 60% sweet spot
    # splitting the in-band region, so the policy can condition on the same
    # boundaries the reward changes at.
    "humidity": [50, 60, 70],
    "soc": [15, 30, 60],
    # Lower edge is the day/night cutoff and is kept equal to the reward's
    # daylight threshold so the policy can condition on the same boundary the
    # energy cost switches on (see compute_reward).
    "solar": [_settings.CLIMATE_SOLAR_DAYLIGHT_W, 100],
    "forecast": [20, 30],
}


def discretize(obs: ClimateObservation) -> tuple[int, ...]:
    return (
        bisect_right(BIN_EDGES["temp"], obs.air_temp_c),
        bisect_right(BIN_EDGES["humidity"], obs.humidity_pct),
        bisect_right(BIN_EDGES["soc"], obs.soc_pct),
        bisect_right(BIN_EDGES["solar"], obs.solar_power_w),
        bisect_right(BIN_EDGES["forecast"], obs.forecast_high_c),
    )


def state_key(obs: ClimateObservation) -> str:
    return str(discretize(obs))


# ---------------------------------------------------------------------------
# Safety rails
# ---------------------------------------------------------------------------

def safety_override(obs: ClimateObservation) -> ClimateAction | None:
    if obs.soc_pct < _settings.CLIMATE_SAFETY_SOC_MIN:
        return ClimateAction(fan=FanAction(percentage=0), mist=False)
    if obs.air_temp_c >= _settings.CLIMATE_SAFETY_TEMP_MAX:
        # Heat emergency: throw everything at it — max fan plus mist. The fan
        # alone can't hold peak afternoons, so the mister is enlisted here. The
        # mist is still subject to apply_mist_safety (off if already saturated).
        return ClimateAction(fan=FanAction(percentage=100), mist=True)
    return None


def apply_mist_safety(obs: ClimateObservation, action: ClimateAction) -> ClimateAction:
    """Force the mister off above the humidity ceiling (fungal / condensation guard).

    There is no humidity term in the reward yet, so this hard rail is what stops
    the mister from saturating the air. Applied to whatever action was chosen,
    including safety overrides.
    """
    if action.mist and obs.humidity_pct >= _settings.CLIMATE_SAFETY_HUMIDITY_MAX:
        return ClimateAction(fan=action.fan, mist=False)
    return action


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

def compute_reward(obs: ClimateObservation, action: ClimateAction) -> float:
    temp = obs.air_temp_c
    temp_min = _settings.CLIMATE_TEMP_MIN_C
    temp_max = _settings.CLIMATE_TEMP_MAX_C
    comfort = -(max(0, temp_min - temp, temp - temp_max)) ** 2

    # Humidity comfort: zero inside the band, quadratic penalty outside. Gives the
    # mister a reason to humidify dry air and to avoid over-saturating, weighted
    # well below temperature so it stays a secondary objective.
    hum = obs.humidity_pct
    hum_min = _settings.CLIMATE_HUMIDITY_MIN_PCT
    hum_max = _settings.CLIMATE_HUMIDITY_MAX_PCT
    humidity = -(max(0, hum_min - hum, hum - hum_max)) ** 2

    # Energy only becomes expensive as the battery drains toward its safety
    # floor. While SOC is healthy (solar keeps it topped up) the fan is treated
    # as nearly free, so we don't suppress cooling the greenhouse actually needs.
    soc_min = _settings.CLIMATE_SAFETY_SOC_MIN
    soc_comfort = _settings.CLIMATE_SOC_COMFORT
    scarcity = (soc_comfort - obs.soc_pct) / (soc_comfort - soc_min)
    # Energy is only truly cheap when there is sun to recharge what the fan
    # spends. At night (solar ~0) there is no recharge, so even a full battery
    # pays a real cost via a higher floor; daytime keeps the low floor.
    daylight = obs.solar_power_w > _settings.CLIMATE_SOLAR_DAYLIGHT_W
    floor = _settings.CLIMATE_ENERGY_FLOOR if daylight else _settings.CLIMATE_ENERGY_FLOOR_NIGHT
    scarcity = min(1.0, max(floor, scarcity))
    energy = -(action.fan_percentage / 100.0) * scarcity

    # Water cost: a flat penalty whenever the mister runs (limited reservoir).
    water = -1.0 if action.mist else 0.0

    return (
        _settings.CLIMATE_REWARD_COMFORT_WEIGHT * comfort
        + _settings.CLIMATE_REWARD_HUMIDITY_WEIGHT * humidity
        + _settings.CLIMATE_REWARD_ENERGY_WEIGHT * energy
        + _settings.CLIMATE_REWARD_WATER_WEIGHT * water
    )


# ---------------------------------------------------------------------------
# Climate policy (wraps generic QLearner)
# ---------------------------------------------------------------------------

class ClimatePolicy:
    name = "q_learning_v2"

    def __init__(self, model_path: str | None = None, **kwargs):
        # Actions are (fan%, mist) pairs. Mist-off block first so a Q-table saved
        # under the old fan-only action space migrates cleanly: old index i (fan
        # level i) maps onto the same index here, and the mist-on actions (5-9)
        # start fresh (QLearner pads short rows on load).
        self.actions = [(fan, mist) for mist in MIST_LEVELS for fan in FAN_LEVELS]
        self._q = QLearner(
            n_actions=len(self.actions),
            model_path=model_path or _settings.CLIMATE_MODEL_PATH,
            **kwargs,
        )
        if not self._q:
            self._warm_start()

    def decide(self, obs: ClimateObservation) -> tuple[ClimateAction, str]:
        idx, explored = self._q.choose(state_key(obs))
        fan_pct, mist = self.actions[idx]
        action = ClimateAction(fan=FanAction(percentage=fan_pct), mist=mist)
        return action, "explore" if explored else "exploit"

    def learn(self, obs: ClimateObservation, action_idx: int, reward: float, next_obs: ClimateObservation) -> None:
        self._q.update(state_key(obs), action_idx, reward, state_key(next_obs))
        self._q.save()

    def action_index(self, action: ClimateAction) -> int:
        closest_fan = min(FAN_LEVELS, key=lambda a: abs(a - action.fan_percentage))
        return self.actions.index((closest_fan, bool(action.mist)))

    def _warm_start(self) -> None:
        logger.info("Warm-starting Q-table with heuristic values")
        bin_counts = [len(e) + 1 for e in BIN_EDGES.values()]
        for combo in itertools.product(*(range(n) for n in bin_counts)):
            t, h, s, *_ = combo
            q = [0.0] * len(self.actions)
            for i, (pct, mist) in enumerate(self.actions):
                frac = pct / 100.0
                if t >= 3:
                    q[i] += frac
                elif t <= 1:
                    q[i] += 1.0 - frac
                if h >= 3:           # too humid (>75%) -> vent it out
                    q[i] += 0.3 * frac
                if s <= 1:
                    q[i] -= 0.5 * frac
                # Mist priors: cools when hot, humidifies dry air, but avoid
                # adding water when it's cold or already humid.
                if mist:
                    if t >= 3:
                        q[i] += 0.5
                    if t <= 1:
                        q[i] -= 0.5
                    if h == 0:       # too dry (<45%) -> mist humidifies
                        q[i] += 0.3
                    if h >= 2:       # at/above comfort top -> avoid over-misting
                        q[i] -= 0.5
            self._q.seed(str(combo), q)
