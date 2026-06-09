from fastapi import FastAPI, HTTPException, Query
from sqlalchemy import select

from demeter import settings as _settings
from demeter.db import get_session, init_db
from demeter.models import DecisionLog, SolarState
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
        return {
            "soc_percent": state.soc_percent,
            "energy_wh": round(state.current_wh, 1),
            "capacity_wh": _settings.BATTERY_CAPACITY_WH,
            "last_updated": state.last_updated.isoformat(),
        }


@app.get("/climate/decisions")
async def climate_decisions(limit: int = Query(default=200, ge=1, le=2000)):
    """Recent climate decisions for offline policy analysis (most recent first)."""
    with get_session() as session:
        rows = session.scalars(
            select(DecisionLog).order_by(DecisionLog.id.desc()).limit(limit)
        ).all()
        decisions = [row.to_api_dict() for row in rows]

    return {"count": len(decisions), "decisions": decisions}


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
