import asyncio
import time

from temporalio import activity

def do_something(obj_id):
    time.sleep(1)
    return "Snap!"

class PlantSnapshotActvities:
    @activity.defn
    async def take_snapshot(self, obj_id) -> str:
        snapshot_result = await asyncio.to_thread(
            do_something, obj_id
        )
        return snapshot_result
