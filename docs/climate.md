# Climate control (Q-learning fan policy)

How demeter decides the greenhouse fan setting. The learner is a tabular
Q-learner (`demeter/qlearning.py`); the greenhouse-specific state, actions,
reward, and safety rails live in `demeter/climate.py`. All tunables are in
`demeter/settings/settings.py` (`CLIMATE_*`).

This doc is the human-readable source of truth for the **state bins** and the
**reward policy**. If you change `BIN_EDGES` or `compute_reward`, update this.

## Observation → state

Each tick observes: average air temp, humidity, battery SOC, solar power,
forecast high. The continuous observation is discretized into a tuple of bin
indices (`discretize`), then stringified as the Q-table key (`state_key`).

Bins (`climate.BIN_EDGES`, via `bisect_right` — edges are lower-inclusive
boundaries):

| feature   | edges          | bins | meaning of each bin                                   |
|-----------|----------------|------|-------------------------------------------------------|
| temp °C   | `13, 20, 28, 35` | 5  | `<13` / `13–20` / `20–28` / `28–35` / `≥35`           |
| humidity %| `40, 65, 85`   | 4    | `<40` / `40–65` / `65–85` / `≥85`                     |
| SOC %     | `15, 30, 60`   | 4    | `<15` / `15–30` / `30–60` / `≥60`                     |
| solar W   | `10, 100`      | 3    | `<10` (night) / `10–100` (day) / `≥100` (strong gen)  |
| forecast °C | `20, 30`     | 3    | `<20` / `20–30` / `≥30`                               |

State space = 5 × 4 × 4 × 3 × 3 = **720** cells.

Notes / known sharp edges:
- **temp** edges at 13 and 28 align with the comfort band (where the reward
  changes); 20 adds in-band resolution; 35 splits the hot zone before the 38°C
  safety cutoff.
- **solar** lower edge (`10`) is kept equal to `CLIMATE_SOLAR_DAYLIGHT_W` so the
  policy can condition on the same day/night boundary the reward's energy cost
  switches on. Keep them equal. Bin 2 (`≥100`) is rarely visited while the
  battery sits full (the controller throttles solar in absorption), so solar is
  effectively night-vs-day in practice.
- **humidity** currently carries **no reward weight** — it inflates the table
  4× but does nothing for the objective yet. It stays for the planned misting
  actuator, which adds a humidity comfort term; realign these edges to the
  comfort band (`~45, 60, 75`) as part of that work.
- **SOC** scarcity in the reward ramps over 15–40%, but the bins break at
  15/30/60. Within bin 2 (30–60%) the policy can't see the 40% scarcity
  boundary. Not urgent (SOC sits ~98% almost always), but add a 40 edge if the
  policy doesn't conserve smoothly on a low-battery stretch.

## Actions

Fan percentage, discretized to `FAN_LEVELS = [0, 25, 50, 75, 100]` (5 actions).

## Reward

`compute_reward(obs, action)` = `W_comfort · comfort + W_energy · energy`.

### Comfort
```
comfort = −( max(0, TEMP_MIN − temp, temp − TEMP_MAX) )²
```
Zero inside the comfort band `[CLIMATE_TEMP_MIN_C, CLIMATE_TEMP_MAX_C]` =
`[13, 28]°C`; a steep quadratic penalty for how far temp strays outside it
(e.g. −25 at 33°C, −81 at 37°C). This term dominates when out of band.

### Energy
```
soc_scarcity = (SOC_COMFORT − soc) / (SOC_COMFORT − SAFETY_SOC_MIN)
floor        = ENERGY_FLOOR        if solar > SOLAR_DAYLIGHT_W   (daytime)
               ENERGY_FLOOR_NIGHT  otherwise                     (night)
scarcity     = clamp(soc_scarcity, floor, 1.0)
energy       = −(fan_percentage / 100) · scarcity
```
The fan's energy cost scales with how scarce battery energy is:
- **High SOC, daytime** → scarcity at the low floor (`0.1`) → fan ~free, so the
  policy isn't discouraged from cooling the greenhouse actually needs. (The fan
  measurably cools — ~1.8 °C/hr at full speed, controlling for temperature.)
- **High SOC, night** → floor rises to `ENERGY_FLOOR_NIGHT` (`0.5`). There is no
  solar to recharge what the fan spends, so running it pointlessly while in-band
  overnight is penalized — without re-suppressing genuinely useful hot-night
  venting (comfort dominates when out of band).
- **Low SOC (toward 15%)** → scarcity → 1.0 regardless of daylight; conserve.

### Weights / current values
| setting | value | role |
|---|---|---|
| `CLIMATE_REWARD_COMFORT_WEIGHT` | 1.0 | comfort term weight |
| `CLIMATE_REWARD_ENERGY_WEIGHT`  | 0.3 | energy term weight |
| `CLIMATE_TEMP_MIN_C` / `_MAX_C` | 13 / 28 | comfort band |
| `CLIMATE_SOC_COMFORT`           | 40 | SOC at/above which energy is cheap |
| `CLIMATE_ENERGY_FLOOR`          | 0.1 | daytime energy-cost floor |
| `CLIMATE_ENERGY_FLOOR_NIGHT`    | 0.5 | night energy-cost floor |
| `CLIMATE_SOLAR_DAYLIGHT_W`      | 10 | day/night cutoff (= solar bin edge) |

## Safety rails

Checked before the policy (`safety_override`), bypassing Q-learning:
- `soc < CLIMATE_SAFETY_SOC_MIN` (`15%`) → fan **off** (protect the battery).
- `air_temp ≥ CLIMATE_SAFETY_TEMP_MAX` (`38°C`) → fan **100%** (protect plants).

## Warm start

On an empty Q-table, `ClimatePolicy._warm_start` seeds heuristic values (hot →
favor fan, cold → favor off, humid → more fan, low solar → penalize fan) so the
policy is sane before it has learned anything.

## Planned (not yet implemented)

- **Misting actuator** (ultrasonic foggers): adds a humidity comfort term to the
  reward and a mist action to the policy; realign humidity bins then. Mist draws
  negligible power, so it is regulated by the humidity band, not an energy cost.
