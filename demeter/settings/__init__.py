import contextlib
import os

from .settings import *

with contextlib.suppress(ImportError):
    from .settings_overrides import *

# Environment variable overrides (useful for Docker)
if _val := os.environ.get("TEMPORAL_HOST"):
    TEMPORAL_HOST = _val
if _val := os.environ.get("SOLAR_DB_PATH"):
    SOLAR_DB_PATH = _val
if _val := os.environ.get("HA_URL"):
    HA_URL = _val
if _val := os.environ.get("HA_TOKEN"):
    HA_TOKEN = _val
if _val := os.environ.get("HA_ENTITY_HUMIDITY"):
    HA_ENTITY_HUMIDITY = _val
if _val := os.environ.get("HA_ENTITY_WEATHER_FORECAST"):
    HA_ENTITY_WEATHER_FORECAST = _val
if _val := os.environ.get("CLIMATE_POLL_INTERVAL_S"):
    CLIMATE_POLL_INTERVAL_S = int(_val)
if _val := os.environ.get("CLIMATE_MODEL_PATH"):
    CLIMATE_MODEL_PATH = _val
