# Climate control (Q-learning fan + mister policy)

How demeter decides the greenhouse fan and mister settings. The learner is a tabular
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
| humidity %| `50, 60, 70`   | 4    | `<50` dry / `50–60` / `60–70` / `≥70` humid           |
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
- **humidity** edges align with the comfort band (50/70, where the crops are
  happy), with 60% (the sweet spot) splitting the in-band region — so the policy
  can condition on the same boundaries the humidity reward changes at.
- **SOC** scarcity in the reward ramps over 15–40%, but the bins break at
  15/30/60. Within bin 2 (30–60%) the policy can't see the 40% scarcity
  boundary. Not urgent (SOC sits ~98% almost always), but add a 40 edge if the
  policy doesn't conserve smoothly on a low-battery stretch.

## Actions

The cross product of fan and mister: `FAN_LEVELS = [0, 25, 50, 75, 100]` ×
mist `{off, on}` = **10 actions**. Ordered mist-off block (indices 0–4) then
mist-on block (5–9), so a Q-table saved under the old fan-only space migrates
cleanly (old fan index → same index; `QLearner` pads short rows on load). The
mister is an on/off switch entity (`HA_ENTITY_MISTER`); a pond fogger ~16 W.

## Reward

`compute_reward(obs, action)` =
`W_comfort · comfort + W_humidity · humidity + W_energy · energy + W_water · water`.

### Comfort (temperature) — asymmetric, too-hot only
```
comfort = −( max(0, temp − TEMP_MAX) )²        # penalty only above 28°C
```
Penalises **too hot** only (e.g. −25 at 33°C, −81 at 37°C); dominates when hot.
**Cold is not penalised** — the fan/mister can cool but nothing can warm, so a
cold penalty was an uncontrollable cost that muddied the cold-state values and
never motivated a useful action (it was driving wasteful cold-night venting).
Same principle as the asymmetric humidity reward: penalise only the directions
the actuators can fix — **hot** (cool) and **dry** (humidify), not **cold**
(no heater) or **humid** (no dehumidifier). Re-add a cold term if a heater is
added (roadmap #4).

### Humidity comfort (asymmetric — too-dry only)
```
humidity = −( max(0, HUMIDITY_MIN − rh) )²      # penalty only below 50% RH
```
Penalises **too dry** only. The mister can *add* humidity, but nothing in the
greenhouse can *remove* it — the fan can't dehumidify when the outside air is
also humid (measured: venting a rainy-day greenhouse left RH flat at +0.04%/tick
while draining the battery). So penalising high RH just bought wasteful venting;
we dropped it. Over-misting into high RH is instead held back by the **water
cost** and the **90% safety rail**. Weighted well below temperature
(`W_humidity = 0.05` vs `W_comfort = 1.0`) so plant temp stays primary.

The fan is therefore driven by **temperature only**; the **mister** is the sole
humidity actuator (humidify when dry). Making the fan a *conditional*
dehumidifier would need outdoor-humidity in the state — deferred to the world
model, since it adds a state dimension for a marginal, hard-to-learn effect.

### Water (mister actuation cost)
```
water = −1.0 if mist else 0.0          # flat per-tick cost when the mister runs
```
Water is a limited, non-replenishing reservoir, so running the mister carries a
flat cost (the mister analogue of the fan's energy term, but it doesn't recharge
with daylight). This gates misting to when the humidity/cooling benefit is worth
the water. `W_water = 0.3` — like the energy floor, too high suppresses useful
misting, so tune against data.

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
| `CLIMATE_REWARD_COMFORT_WEIGHT` | 1.0 | temperature comfort weight |
| `CLIMATE_REWARD_HUMIDITY_WEIGHT`| 0.05 | humidity comfort weight (tuning knob) |
| `CLIMATE_REWARD_ENERGY_WEIGHT`  | 0.3 | fan energy term weight |
| `CLIMATE_REWARD_WATER_WEIGHT`   | 0.3 | mister water cost (tuning knob) |
| `CLIMATE_TEMP_MIN_C` / `_MAX_C` | 13 / 28 | temperature comfort band |
| `CLIMATE_HUMIDITY_MIN_PCT` / `_MAX_PCT` | 50 / 70 | humidity comfort band |
| `CLIMATE_SOC_COMFORT`           | 40 | SOC at/above which energy is cheap |
| `CLIMATE_ENERGY_FLOOR`          | 0.1 | daytime energy-cost floor |
| `CLIMATE_ENERGY_FLOOR_NIGHT`    | 0.5 | night energy-cost floor |
| `CLIMATE_SOLAR_DAYLIGHT_W`      | 10 | day/night cutoff (= solar bin edge) |

## Safety rails

Full overrides checked before the policy (`safety_override`), bypassing Q-learning:
- `soc < CLIMATE_SAFETY_SOC_MIN` (`15%`) → fan **off**, mist **off** (protect the battery).
- `air_temp ≥ CLIMATE_SAFETY_TEMP_MAX` (`38°C`) → fan **100%** + mist **on** — heat
  emergency throws all cooling at it, since the fan alone can't hold peak afternoons.

A separate clamp (`apply_mist_safety`) is applied to whatever action is finally
chosen (policy or override):
- `humidity ≥ CLIMATE_SAFETY_HUMIDITY_MAX` (`90%`) → mist forced **off**
  (fungal / condensation guard). This is the hard backstop on over-humidifying.

## Warm start

On an empty Q-table, `ClimatePolicy._warm_start` seeds heuristic values so the
policy is sane before it has learned anything: hot → favor fan, cold → favor off,
too dry → favor mist, hot → favor mist (it cools), already-humid → avoid mist,
low solar → penalize fan. (The fan has no humidity prior — it's temperature-only.)

## Notes / future

- **Action-space size.** 10 actions doubles the exploration needed per state vs
  the old 5, which worsens the (already poor) hot-state sample efficiency — hot
  states are rare, so the fan/mist response there learns slowly. See below.
- **World model (roadmap).** Model-free Q-learning struggles to learn the
  fan/mister cooling effect in rare, transient hot states (the signal is small
  and confounded with exogenous heat). A learned thermal dynamics model would
  let the controller *reason* about cooling instead of discovering it through
  noisy trial-and-error — the principled fix for the hot regime.
