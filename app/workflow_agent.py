from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
import asyncio

class MyWorkflow(Workflow):
    @step
    async def my_step(self, ev: StartEvent) -> StopEvent:
        return StopEvent(result="Hello, world!")


async def main():
    w = MyWorkflow(timeout=10, verbose=False)
    result = await w.run()
    print(result)

asyncio.run(main())
