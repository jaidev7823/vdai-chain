from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
import asyncio
from llama_index.utils.workflow import draw_all_possible_flows


class MyWorkflow(Workflow):
    @step
    async def my_step(self, ev: StartEvent) -> StopEvent:
        return StopEvent(result="Hello, world!")

draw_all_possible_flows(
    MyWorkflow,
    filename="basic_workflow.html",
    # Optional, can limit long event names in your workflow
    # Can help with readability
    # max_label_length=10,
)

async def main():
    w = MyWorkflow(timeout=10, verbose=False)
    result = await w.run()
    print(result)

asyncio.run(main())
