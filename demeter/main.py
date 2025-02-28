from fastapi import FastAPI, HTTPException
from reolink import Host

app = FastAPI()


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
