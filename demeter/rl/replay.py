"""Offline experience replay for the climate Q-learner.

Rebuilds the Q-table from the DecisionLog instead of relying solely on the live
2-min-cadence TD updates. Each pair of consecutive log rows is one transition
(s = obs_t, a = action_t, s' = obs_t+1); we recompute the reward from the raw
observations under the *current* reward function and sweep the whole log for N
epochs.

Because the log stores raw observations and actions (not discretized state keys
or action indices), replay is robust to redesign:
  - change bin edges        -> obs are re-discretized with the current edges
  - add an action           -> no old samples; filled in by live exploration
  - remove an action        -> old samples for it are dropped (or --snap'd)
  - retune the reward        -> rewards are recomputed here, no new data needed
The Q-table is therefore a disposable cache; the DecisionLog is the source of
truth. Delete the table, replay, and you have a fresh, correct policy.

Run as a manual script:  python -m demeter.rl.replay [--epochs N] [--dry-run] ...
"""

import argparse
import json
import logging
import random
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

try:
    from demeter import settings as _settings
    from demeter.rl.climate import (
        ClimateAction,
        ClimateObservation,
        ClimatePolicy,
        FanAction,
        compute_reward,
        state_key,
    )
    from demeter.db import get_session
    from demeter.models import DecisionLog
except ImportError:
    import settings as _settings
    from rl.climate import (
        ClimateAction,
        ClimateObservation,
        ClimatePolicy,
        FanAction,
        compute_reward,
        state_key,
    )
    from db import get_session
    from models import DecisionLog

from sqlalchemy import select

logger = logging.getLogger(__name__)

# Passed as the policy's model_path so QLearner._load finds no file and the
# policy warm-starts from heuristics instead of loading the live table. We
# rebuild from scratch and only write to the real path at the very end.
_NO_LOAD_SENTINEL = "__replay_rebuild_no_load__"


@dataclass(frozen=True)
class Transition:
    obs: ClimateObservation
    action: ClimateAction
    next_obs: ClimateObservation


def _parse_obs(obs_json: str) -> ClimateObservation:
    return ClimateObservation(**json.loads(obs_json))


def _parse_action(action_json: str) -> ClimateAction:
    d = json.loads(action_json)
    fan = d.get("fan") or {}
    return ClimateAction(
        fan=FanAction(percentage=fan.get("percentage", 0)),
        mist=bool(d.get("mist", False)),
    )


def load_rows() -> list[tuple[str, str]]:
    """All (observation_json, action_json) from the DecisionLog, in time order."""
    with get_session() as session:
        rows = session.execute(
            select(DecisionLog.observation_json, DecisionLog.action_json).order_by(DecisionLog.id)
        ).all()
    return [(o, a) for o, a in rows]


def build_transitions(rows: list[tuple[str, str]], max_gap_s: float) -> list[Transition]:
    """Pair consecutive rows into (s, a, s') transitions.

    Drops a pair when the wall-clock gap between the two observations is <=0 or
    larger than max_gap_s — that means the workflow was down, so s' is not
    actually the consequence of a. A row that fails to parse becomes a None
    placeholder so we never pair *across* it.
    """
    parsed: list[tuple[ClimateObservation, ClimateAction] | None] = []
    for obs_json, action_json in rows:
        try:
            parsed.append((_parse_obs(obs_json), _parse_action(action_json)))
        except Exception as e:
            logger.warning("Skipping unparseable log row: %s", e)
            parsed.append(None)

    transitions: list[Transition] = []
    for cur, nxt in zip(parsed, parsed[1:]):
        if cur is None or nxt is None:
            continue
        obs, action = cur
        next_obs, _ = nxt
        try:
            gap = (
                datetime.fromisoformat(next_obs.timestamp)
                - datetime.fromisoformat(obs.timestamp)
            ).total_seconds()
        except ValueError:
            continue
        if gap <= 0 or gap > max_gap_s:
            continue
        transitions.append(Transition(obs, action, next_obs))
    return transitions


def _resolve_index(policy: ClimatePolicy, action: ClimateAction, snap: bool) -> int | None:
    """Index of the executed action in the current action space, or None to drop."""
    key = (action.fan_percentage, bool(action.mist))
    if key in policy.actions:
        return policy.actions.index(key)
    if snap:
        return policy.action_index(action)  # snaps fan% to the nearest level
    return None


def prepare(
    policy: ClimatePolicy, transitions: list[Transition], snap: bool
) -> tuple[list[tuple[str, int, float, str]], int]:
    """Turn transitions into (state_key, action_idx, reward, next_state_key) samples.

    Reward is recomputed from the raw next observation under the current reward
    function — the same way the live loop does it (compute_reward(next_obs, a)).
    The reward uses the *executed* action (real energy cost), while the index may
    be snapped; returns (samples, dropped_count).
    """
    samples: list[tuple[str, int, float, str]] = []
    dropped = 0
    for t in transitions:
        idx = _resolve_index(policy, t.action, snap)
        if idx is None:
            dropped += 1
            continue
        reward = compute_reward(t.next_obs, t.action)
        samples.append((state_key(t.obs), idx, reward, state_key(t.next_obs)))
    return samples, dropped


def run_replay(
    policy: ClimatePolicy,
    samples: list[tuple[str, int, float, str]],
    epochs: int,
    shuffle: bool = True,
) -> None:
    """Sweep the samples through the Q-learner for `epochs` passes.

    Epsilon is restored afterwards: update() decays it per call, but replay is
    not real experience, so the live system should resume with its original
    exploration rate rather than a value decayed by tens of thousands of replays.
    """
    q = policy._q
    eps0 = q.epsilon
    for _ in range(epochs):
        order = list(samples)
        if shuffle:
            random.shuffle(order)
        for s, a, r, s_next in order:
            q.update(s, a, r, s_next)
    q.epsilon = eps0


def rebuild(
    epochs: int,
    max_gap_s: float,
    snap: bool,
    alpha: float | None,
    output: str,
    dry_run: bool,
) -> ClimatePolicy:
    rows = load_rows()
    transitions = build_transitions(rows, max_gap_s)

    # Fresh, warm-started policy that does NOT load the live table.
    policy = ClimatePolicy(model_path=_NO_LOAD_SENTINEL)
    if alpha is not None:
        policy._q.alpha = alpha

    samples, dropped = prepare(policy, transitions, snap)
    run_replay(policy, samples, epochs)

    states = len({s for s, _, _, _ in samples})
    mean_r = sum(r for _, _, r, _ in samples) / len(samples) if samples else 0.0
    logger.info(
        "rows=%d transitions=%d used=%d dropped=%d states_covered=%d "
        "mean_reward=%.3f epochs=%d q_states=%d",
        len(rows), len(transitions), len(samples), dropped, states, mean_r, epochs, len(policy._q),
    )

    if dry_run:
        logger.info("dry-run: not writing Q-table")
        return policy

    out = Path(output)
    if out.exists():
        backup = out.with_suffix(out.suffix + ".bak")
        shutil.copy2(out, backup)
        logger.info("backed up existing Q-table to %s", backup)
    policy._q.model_path = output
    policy._q.save()
    logger.info("wrote rebuilt Q-table to %s (%d states)", output, len(policy._q))
    return policy


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Rebuild the climate Q-table from the DecisionLog.")
    ap.add_argument("--epochs", type=int, default=20, help="passes over the log (default 20)")
    ap.add_argument("--alpha", type=float, default=None, help="override learning rate for replay")
    ap.add_argument(
        "--max-gap-s",
        type=float,
        default=2 * _settings.CLIMATE_POLL_INTERVAL_S,
        help="drop transitions whose obs are more than this many seconds apart",
    )
    ap.add_argument(
        "--snap",
        action="store_true",
        help="map removed/unknown actions to the nearest fan level instead of dropping them",
    )
    ap.add_argument("--output", default=_settings.CLIMATE_MODEL_PATH, help="where to write the Q-table")
    ap.add_argument("--dry-run", action="store_true", help="compute and report stats but do not write")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for epoch shuffling")
    args = ap.parse_args()

    random.seed(args.seed)
    rebuild(
        epochs=args.epochs,
        max_gap_s=args.max_gap_s,
        snap=args.snap,
        alpha=args.alpha,
        output=args.output,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
