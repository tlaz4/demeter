from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from activities import PlantSnapshotActvities

@workflow.defn
class PlantSnapshot:
    @workflow.run
    async def run(self, obj_id: str) -> str:
        retry_policy = RetryPolicy(
            maximum_attempts=3,
            maximum_interval=timedelta(seconds=2)
        )

        take_snapshot_output = await workflow.execute_activity_method(
            PlantSnapshotActvities.take_snapshot,
            obj_id,
            start_to_close_timeout=timedelta(seconds=5),
            retry_policy=retry_policy
        )

        result = f"Transfer complete"
        return result
