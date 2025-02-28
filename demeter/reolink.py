import xml.etree.ElementTree as ET

from demeter.reolink_aio.reolink_aio.baichuan import Baichuan
import asyncio

class Host:
    def __init__(self):
        self.host = Baichuan('192.168.1.78', 'admin', 'TyWater1995!!2')

    async def login(self):
        await self.host.login()

    async def get_pan_tilt(self):
        await self.host.get_ptz_position(0)

        return self.host.pan_position(0), self.host.tilt_position(0)

    async def get_ptz_presets(self):
        message = await self.host.send(cmd_id=190, channel=0)
        return message

    async def move_to_ptz_preset(self, preset, id):
        preset_xml = f"""<?xml version="1.0" encoding="UTF-8" ?>
        <body>
        <PtzPreset version="1.1">
        <channelId>0</channelId>
        <presetList>
        <preset>
        <id>{id}</id>
        <command>toPos</command>
        <name>{preset}</name>
        </preset>
        </presetList>
        </PtzPreset>
        </body>
        """

        await self.host.send(cmd_id=19, channel=0, body=preset_xml)

    async def calibrate(self):
        await self.host.send(cmd_id=341, channel=0)

    @classmethod
    def parse_preset_xml(cls, xml_body):
        root = ET.fromstring(xml_body)
        presets = root.findall(".//preset")

        return {
            preset.find("name").text: preset.find("id").text for preset in presets
        }




async def main():
    host = Host()
    await host.login()

    pan, tilt = await host.get_pan_tilt()
    print(pan, tilt)

    ptz_presets = await host.get_ptz_presets()
    print(ptz_presets)

    print(Host.parse_preset_xml(ptz_presets))

    # await host.move_to_ptz_preset("seedlings", 1)

if __name__ == "__main__":
    asyncio.run(main())