import json
import logging
import random
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)


class QLearner:
    """Tabular Q-learning with epsilon-greedy exploration and JSON persistence."""

    def __init__(
        self,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 0.15,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.9995,
        model_path: str | None = None,
    ):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model_path = model_path
        self._q: dict[str, list[float]] = defaultdict(lambda: [0.0] * n_actions)

        if model_path:
            self._load()

    def __bool__(self) -> bool:
        return bool(self._q)

    def __len__(self) -> int:
        return len(self._q)

    def choose(self, state: str) -> tuple[int, bool]:
        """Pick an action index. Returns (action_idx, explored)."""
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions), True
        q = self._q[state]
        return max(range(self.n_actions), key=lambda i: q[i]), False

    def update(self, state: str, action: int, reward: float, next_state: str) -> None:
        """TD(0) update and epsilon decay."""
        old = self._q[state][action]
        best_next = max(self._q[next_state])
        self._q[state][action] = old + self.alpha * (reward + self.gamma * best_next - old)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def seed(self, state: str, values: list[float]) -> None:
        """Set Q-values for a state directly (for warm-starting)."""
        self._q[state] = values

    def save(self) -> None:
        if not self.model_path:
            return
        try:
            path = Path(self.model_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"q": dict(self._q), "epsilon": self.epsilon}))
        except Exception as e:
            logger.warning("Failed to save Q-table: %s", e)

    def _load(self) -> None:
        path = Path(self.model_path)
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
            migrated = 0
            for k, v in data["q"].items():
                if len(v) < self.n_actions:
                    # Action space grew (e.g. a new actuator) — keep the learned
                    # values for existing actions, start new ones at zero.
                    v = v + [0.0] * (self.n_actions - len(v))
                    migrated += 1
                elif len(v) > self.n_actions:
                    v = v[: self.n_actions]
                    migrated += 1
                self._q[k] = v
            self.epsilon = data.get("epsilon", self.epsilon)
            if migrated:
                logger.info("Migrated %d Q-rows to n_actions=%d", migrated, self.n_actions)
            logger.info("Loaded Q-table: %d states, epsilon=%.4f", len(self._q), self.epsilon)
        except Exception as e:
            logger.warning("Failed to load Q-table: %s", e)
