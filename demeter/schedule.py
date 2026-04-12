import asyncio
import logging
from datetime import timedelta

from temporalio.client import (
    Client,
    Schedule,
    ScheduleActionStartWorkflow,
    ScheduleAlreadyRunningError,
    ScheduleIntervalSpec,
    ScheduleOverlapPolicy,
    SchedulePolicy,
    ScheduleSpec,
    ScheduleState,
    ScheduleUpdate,
)

import settings as _settings
from shared import PLANT_SNAPSHOT_TASK_QUEUE_NAME, SOLAR_POLL_TASK_QUEUE_NAME
from workflows import PlantSnapshot, SolarPoll

logger = logging.getLogger(__name__)

_SCHEDULES = [
    (
        "plant-snapshot-schedule-id",
        Schedule(
            action=ScheduleActionStartWorkflow(
                PlantSnapshot.run,
                "argument",
                id="schedules-plant-snapshot-id",
                task_queue=PLANT_SNAPSHOT_TASK_QUEUE_NAME,
            ),
            spec=ScheduleSpec(
                intervals=[ScheduleIntervalSpec(every=timedelta(minutes=2))]
            ),
            state=ScheduleState(note="Periodic plant snapshot."),
        ),
    ),
    (
        "solar-poll-schedule-id",
        Schedule(
            action=ScheduleActionStartWorkflow(
                SolarPoll.run,
                id="solar-poll-workflow-id",
                task_queue=SOLAR_POLL_TASK_QUEUE_NAME,
            ),
            spec=ScheduleSpec(
                intervals=[ScheduleIntervalSpec(every=timedelta(seconds=_settings.SOLAR_POLL_INTERVAL_S))]
            ),
            policy=SchedulePolicy(overlap=ScheduleOverlapPolicy.SKIP),
            state=ScheduleState(note="Solar battery SOC estimation via coulomb counting."),
        ),
    ),
]


async def register_schedules(client: Client) -> None:
    for schedule_id, schedule in _SCHEDULES:
        try:
            await client.create_schedule(schedule_id, schedule)
            logger.info("Registered schedule: %s", schedule_id)
        except ScheduleAlreadyRunningError:
            handle = client.get_schedule_handle(schedule_id)
            await handle.update(lambda _: ScheduleUpdate(schedule=schedule))
            logger.info("Updated existing schedule: %s", schedule_id)


async def main():
    client = await Client.connect(_settings.TEMPORAL_HOST)
    await register_schedules(client)


if __name__ == "__main__":
    asyncio.run(main())
