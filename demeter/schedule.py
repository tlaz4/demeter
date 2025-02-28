import asyncio
from datetime import timedelta

from temporalio.client import (
    Client,
    Schedule,
    ScheduleActionStartWorkflow,
    ScheduleIntervalSpec,
    ScheduleSpec,
    ScheduleState,
)

from shared import PLANT_SNAPSHOT_TASK_QUEUE_NAME
from workflows import PlantSnapshot


async def main():
    client = await Client.connect("localhost:7233")

    await client.create_schedule(
        "plant-snapshot-schedule-id",
        Schedule(
            action=ScheduleActionStartWorkflow(
                PlantSnapshot.run,
                "argument",
                id="schedules-plant-snapshot-id",
                task_queue=PLANT_SNAPSHOT_TASK_QUEUE_NAME
            ),
            spec=ScheduleSpec(
                intervals=[ScheduleIntervalSpec(every=timedelta(minutes=2))]
            ),
            state=ScheduleState(note="Here's a note on my Schedule.")
        )
    )

if __name__ == "__main__":
    asyncio.run(main())