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
    "humidity": [40, 65, 85],
    "soc": [15, 30, 60],
    "solar": [10, 100],
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
        return ClimateAction(fan=FanAction(percentage=0))
    if obs.air_temp_c >= _settings.CLIMATE_SAFETY_TEMP_MAX:
        return ClimateAction(fan=FanAction(percentage=100))
    return None


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

def compute_reward(obs: ClimateObservation, action: ClimateAction) -> float:
    temp = obs.air_temp_c
    temp_min = _settings.CLIMATE_TEMP_MIN_C
    temp_max = _settings.CLIMATE_TEMP_MAX_C
    comfort = -(max(0, temp_min - temp, temp - temp_max)) ** 2

    # Energy only becomes expensive as the battery drains toward its safety
    # floor. While SOC is healthy (solar keeps it topped up) the fan is treated
    # as nearly free, so we don't suppress cooling the greenhouse actually needs.
    soc_min = _settings.CLIMATE_SAFETY_SOC_MIN
    soc_comfort = _settings.CLIMATE_SOC_COMFORT
    scarcity = (soc_comfort - obs.soc_pct) / (soc_comfort - soc_min)
    scarcity = min(1.0, max(_settings.CLIMATE_ENERGY_FLOOR, scarcity))
    energy = -(action.fan_percentage / 100.0) * scarcity

    return _settings.CLIMATE_REWARD_COMFORT_WEIGHT * comfort + _settings.CLIMATE_REWARD_ENERGY_WEIGHT * energy


# ---------------------------------------------------------------------------
# Climate policy (wraps generic QLearner)
# ---------------------------------------------------------------------------

class ClimatePolicy:
    name = "q_learning_v1"

    def __init__(self, model_path: str | None = None, **kwargs):
        self.actions = FAN_LEVELS
        self._q = QLearner(
            n_actions=len(self.actions),
            model_path=model_path or _settings.CLIMATE_MODEL_PATH,
            **kwargs,
        )
        if not self._q:
            self._warm_start()

    def decide(self, obs: ClimateObservation) -> tuple[ClimateAction, str]:
        idx, explored = self._q.choose(state_key(obs))
        pct = self.actions[idx]
        return ClimateAction(fan=FanAction(percentage=pct)), "explore" if explored else "exploit"

    def learn(self, obs: ClimateObservation, action_idx: int, reward: float, next_obs: ClimateObservation) -> None:
        self._q.update(state_key(obs), action_idx, reward, state_key(next_obs))
        self._q.save()

    def action_index(self, action: ClimateAction) -> int:
        closest = min(self.actions, key=lambda a: abs(a - action.fan_percentage))
        return self.actions.index(closest)

    def _warm_start(self) -> None:
        logger.info("Warm-starting Q-table with heuristic values")
        bin_counts = [len(e) + 1 for e in BIN_EDGES.values()]
        for combo in itertools.product(*(range(n) for n in bin_counts)):
            t, h, s, *_ = combo
            q = [0.0] * len(self.actions)
            for i, pct in enumerate(self.actions):
                frac = pct / 100.0
                if t >= 3:
                    q[i] += frac
                elif t <= 1:
                    q[i] += 1.0 - frac
                if h >= 2:
                    q[i] += 0.3 * frac
                if s <= 1:
                    q[i] -= 0.5 * frac
            self._q.seed(str(combo), q)
