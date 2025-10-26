from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Event,
)
import asyncio
from llama_index.utils.workflow import draw_all_possible_flows
import random

class FirstEvent(Event):
    first_output: str

class SecondEvent(Event):
    second_output: str

class LoopEvent(Event):
    loop_output: str

class MyWorkflow(Workflow):
    @step
    async def step_one(self, ev: StartEvent | LoopEvent) -> FirstEvent | LoopEvent:
        if random.randint(0, 1) == 0:
            print("Bad thing happened")
            return LoopEvent(loop_output="Back to step one.")
        else:
            print("Good thing happened")
            return FirstEvent(first_output="First step complete.")

    @step
    async def step_two(self, ev: FirstEvent) -> SecondEvent:
        print(ev.first_output)
        return SecondEvent(second_output="Second step complete.")

    @step
    async def step_three(self, ev: SecondEvent) -> StopEvent:
        print(ev.second_output)
        return StopEvent(result="Workflow complete.")

draw_all_possible_flows(
    MyWorkflow,
    filename="basic_workflow.html",
    # Optional, can limit long event names in your workflow
    # Can help with readability
    # max_label_length=10,
)

async def main():
    w = MyWorkflow(timeout=10, verbose=False)
    result = await w.run(first_input="Start the workflow.")
    print(result)

asyncio.run(main())
