import asyncio
import logging

from temporalio.client import Client
from temporalio.worker import Worker

import settings as _settings
from activities import PlantSnapshotActvities, SolarPollActivities
from schedule import register_schedules
from shared import PLANT_SNAPSHOT_TASK_QUEUE_NAME, SOLAR_POLL_TASK_QUEUE_NAME
from workflows import PlantSnapshot, SolarPoll

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


async def main() -> None:
    client: Client = await Client.connect(_settings.TEMPORAL_HOST, namespace="default")

    await register_schedules(client)

    plant_snapshot_activities = PlantSnapshotActvities()
    solar_poll_activities = SolarPollActivities()

    await asyncio.gather(
        Worker(
            client,
            task_queue=PLANT_SNAPSHOT_TASK_QUEUE_NAME,
            workflows=[PlantSnapshot],
            activities=[plant_snapshot_activities.take_snapshot],
        ).run(),
        Worker(
            client,
            task_queue=SOLAR_POLL_TASK_QUEUE_NAME,
            workflows=[SolarPoll],
            activities=[solar_poll_activities.poll_solar],
        ).run(),
    )


if __name__ == "__main__":
    asyncio.run(main())
