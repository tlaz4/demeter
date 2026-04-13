from fastapi import FastAPI, HTTPException

from demeter import settings as _settings
from demeter.db import get_session, init_db
from demeter.models import SolarState
from demeter.reolink import Host

app = FastAPI()

init_db()


@app.get("/status")
async def status():
    return {"status": True}


@app.get("/solar/status")
async def solar_status():
    with get_session() as session:
        state = session.get(SolarState, 1)
        if state is None:
            raise HTTPException(status_code=503, detail="Solar data not yet available")
        soc = round((state.current_wh / _settings.BATTERY_CAPACITY_WH) * 100.0, 1)
        return {
            "soc_percent": soc,
            "energy_wh": round(state.current_wh, 1),
            "capacity_wh": _settings.BATTERY_CAPACITY_WH,
            "last_updated": state.last_updated.isoformat(),
        }


@app.post("/move-to-preset/{preset}")
async def move_to_preset(preset: str):
    host = Host()
    await host.login()

    ptz_presets = await host.get_ptz_presets()
    parsed_presets = Host.parse_preset_xml(ptz_presets)

    preset_id = parsed_presets.get(preset)

    if not preset_id:
        raise HTTPException(status_code=404, detail="Preset not found")

    await host.move_to_ptz_preset(preset, preset_id)
    pan, tilt = await host.get_pan_tilt()

    return {"currentPreset": preset, "pan": pan, "tilt": tilt}
