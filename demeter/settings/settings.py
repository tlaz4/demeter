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
    {"name": "fans", "entity_id": "fan.greenhouse_power_module_fan", "power_w": 24.0, "type": "percentage"},
    {"name": "mppt_load", "entity_id": "sensor.greenhouse_solar_module_load_power", "power_w": 0.0, "type": "sensor"},
]
