import asyncio
import traceback

from temporalio.client import Client, WorkflowFailureError

from shared import PLANT_SNAPSHOT_TASK_QUEUE_NAME
from workflows import PlantSnapshot


async def main() -> None:
    # Create client connected to server at the given address
    client: Client = await Client.connect("localhost:7233")

    try:
        result = await client.execute_workflow(
            PlantSnapshot.run,
            "something",
            id="pay-invoice-701",
            task_queue=PLANT_SNAPSHOT_TASK_QUEUE_NAME,
        )

        print(f"Result: {result}")

    except WorkflowFailureError:
        print("Got expected exception: ", traceback.format_exc())


if __name__ == "__main__":
    asyncio.run(main())