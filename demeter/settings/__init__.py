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
