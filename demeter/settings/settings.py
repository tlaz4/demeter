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

# Known loads
# type "binary"     — on/off entity, power_w is full draw
# type "percentage" — PWM entity, power_w is max draw
# type "sensor"     — HA power sensor reporting actual watts, power_w unused
LOADS = [
    {"name": "fans", "entity_id": "fan.greenhouse_power_module_fan", "power_w": 48.0, "type": "percentage"},
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
CLIMATE_TEMP_MIN_C = 13.0
CLIMATE_TEMP_MAX_C = 28.0
CLIMATE_SAFETY_SOC_MIN = 15.0
CLIMATE_SAFETY_TEMP_MAX = 38.0
CLIMATE_REWARD_COMFORT_WEIGHT = 1.0
CLIMATE_REWARD_ENERGY_WEIGHT = 0.3
# Energy is only "expensive" as SOC drains toward the safety floor. At/above
# CLIMATE_SOC_COMFORT the fan is treated as ~free (solar keeps the battery
# topped up), scaling to full cost at CLIMATE_SAFETY_SOC_MIN. The floor keeps a
# small cost even when full so the agent still prefers minimum effective fan.
CLIMATE_SOC_COMFORT = 40.0
CLIMATE_ENERGY_FLOOR = 0.1
CLIMATE_POLL_INTERVAL_S = 120
CLIMATE_MODEL_PATH = "/data/climate_q.json"
