import asyncio

from temporalio.client import Client
from temporalio.worker import Worker

from activities import PlantSnapshotActvities
from shared import PLANT_SNAPSHOT_TASK_QUEUE_NAME
from workflows import PlantSnapshot


async def main() -> None:
    client: Client = await Client.connect("localhost:7233", namespace="default")
    # Run the worker
    activities = PlantSnapshotActvities()
    worker: Worker = Worker(
        client,
        task_queue=PLANT_SNAPSHOT_TASK_QUEUE_NAME,
        workflows=[PlantSnapshot],
        activities=[activities.take_snapshot],
    )
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())