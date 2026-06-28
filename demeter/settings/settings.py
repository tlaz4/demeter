REOLINK_ADDRESS = "123"
REOLINK_USERNAME = "exampl"
REOLINK_PASSWORD = "abc123"

# Home Assistant
HA_URL = "http://homeassistant.local:8123"
HA_TOKEN = "your_long_lived_token_here"

# MPPT sensor entity IDs (ESPHome via HA)
HA_ENTITY_BATTERY_VOLTAGE = "sensor.greenhouse_solar_module_battery_voltage"
HA_ENTITY_SOLAR_POWER     = "sensor.greenhouse_solar_module_solar_power"
HA_ENTITY_BATTERY_TEMP    = "sensor.greenhouse_solar_module_battery_temperature"

# Demeter-managed sensor pushed back to HA
HA_ENTITY_SOC = "sensor.demeter_battery_soc"

# Temporal
TEMPORAL_HOST = "localhost:7233"

# Battery
BATTERY_CAPACITY_WH = 1200   # 100Ah × 12V
SOLAR_POLL_INTERVAL_S = 60
SOLAR_DB_PATH = "/tmp/demeter.db"

# Pond fogger (mister) — exposed in HA as an on/off switch entity.
HA_ENTITY_MISTER = "switch.greenhouse_power_module_fogger"
CLIMATE_MISTER_POWER_W = 16.0   # pond fogger draw — tracked for SOC, not penalised in the reward

# Known loads
# type "binary"     — on/off entity, power_w is full draw
# type "percentage" — PWM entity, power_w is max draw
# type "sensor"     — HA power sensor reporting actual watts, power_w unused
LOADS = [
    {"name": "fans", "entity_id": "fan.greenhouse_power_module_fan", "power_w": 48.0, "type": "percentage"},
    {"name": "mister", "entity_id": HA_ENTITY_MISTER, "power_w": CLIMATE_MISTER_POWER_W, "type": "binary"},
    {"name": "mppt_load", "entity_id": "sensor.greenhouse_solar_module_load_power", "power_w": 0.0, "type": "sensor"},
]

# Climate control
HA_ENTITY_FAN = "fan.greenhouse_power_module_fan"
HA_ENTITY_AIR_TEMPS = [
    "sensor.greenhouse_sensor1_greenhouse_air_temperature",
    "sensor.greenhouse_sensor1_greenhouse_node_1_temperature",
    "sensor.greenhouse_solar_module_battery_temperature"
]
HA_ENTITY_HUMIDITY = "sensor.greenhouse_sensor1_greenhouse_air_humidity"
HA_ENTITY_WEATHER_FORECAST = "sensor.edmonton_high_temperature"
# Temperature comfort band. The reward only penalises *too hot* (above MAX) —
# nothing can warm the greenhouse, so cold isn't penalised. MIN documents the
# band floor / temp bin edge; re-introduce a cold penalty if a heater is added.
CLIMATE_TEMP_MIN_C = 13.0
CLIMATE_TEMP_MAX_C = 28.0
# Humidity comfort band (%RH) — matches where the crops are happy (50-70).
# The reward only penalises *too dry* (below min): the mister can humidify, but
# nothing can remove humidity, so penalising high RH just drove wasteful fan
# venting. MAX_PCT documents the band top and aligns the bins / mister warm-start.
CLIMATE_HUMIDITY_MIN_PCT = 50.0
CLIMATE_HUMIDITY_MAX_PCT = 70.0
CLIMATE_SAFETY_SOC_MIN = 15.0
CLIMATE_SAFETY_TEMP_MAX = 38.0
# Above this RH the mister is forced off (fungal / condensation guard). This is
# the only brake on over-humidifying until a humidity comfort term is added.
CLIMATE_SAFETY_HUMIDITY_MAX = 90.0
CLIMATE_REWARD_COMFORT_WEIGHT = 1.0
CLIMATE_REWARD_ENERGY_WEIGHT = 0.3
# Humidity comfort weight, kept small relative to temperature: plant temp is the
# primary objective, humidity secondary. This is the key tuning knob — set it too
# high and the immediate humidity penalty drowns out the slow-to-learn cooling
# benefit, suppressing useful misting (the same trap the energy floor had). Tune
# against logged data.
CLIMATE_REWARD_HUMIDITY_WEIGHT = 0.05
# Mister actuation cost. Water is a limited, non-replenishing reservoir, so each
# tick the mister runs carries a flat penalty (unlike fan energy, it doesn't
# recharge with daylight). Gates misting to when the humidity/cooling benefit is
# worth the water. Like the energy floor, too high suppresses useful misting.
CLIMATE_REWARD_WATER_WEIGHT = 0.3
# Energy is only "expensive" as SOC drains toward the safety floor. At/above
# CLIMATE_SOC_COMFORT the fan is treated as ~free (solar keeps the battery
# topped up), scaling to full cost at CLIMATE_SAFETY_SOC_MIN.
CLIMATE_SOC_COMFORT = 40.0
# Per-tick energy cost has a minimum floor so the agent still prefers minimum
# effective fan. The floor is daylight-dependent: in daytime (solar present to
# recharge) it stays low so cooling isn't suppressed; at night (no recharge)
# it rises so the fan isn't run pointlessly, draining the battery until sunrise.
# CLIMATE_SOLAR_DAYLIGHT_W is the day/night cutoff and MUST match the lower
# solar bin edge in climate.BIN_EDGES so the policy can act on the same boundary.
CLIMATE_ENERGY_FLOOR = 0.1
CLIMATE_ENERGY_FLOOR_NIGHT = 0.5
CLIMATE_SOLAR_DAYLIGHT_W = 10.0
CLIMATE_POLL_INTERVAL_S = 120
CLIMATE_MODEL_PATH = "/data/climate_q.json"
