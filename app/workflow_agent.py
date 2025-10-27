from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Event,
    Context,
)
import asyncio
from llama_index.utils.workflow import draw_all_possible_flows
import random

class SetupEvent(Event):
    query: str


class StepTwoEvent(Event):
    query: str


class StatefulFlow(Workflow):
    @step
    async def start(
        self, ctx: Context, ev: StartEvent
    ) -> SetupEvent | StepTwoEvent:
        db = await ctx.store.get("some_database", default=None)
        if db is None:
            print("Need to load data")
            return SetupEvent(query=ev.query)

        # do something with the query
        return StepTwoEvent(query=ev.query)

    @step
    async def setup(self, ctx: Context, ev: SetupEvent) -> StartEvent:
        # load data
        async with ctx.store.edit_state() as state:
            state["some_database"] = [1, 2, 3]
            state["metadata"] = {"initialized": True}
        return StartEvent(query=ev.query)
    
    @step
    async def step_two(self, ctx: Context, ev: StepTwoEvent) -> StopEvent:
        async with ctx.store.edit_state() as state:
            # you can also read+modify existing keys
            state["run_count"] = state.get("run_count", 0) + 1
        db = await ctx.store.get("some_database")
        print("Data is ", db)
        return StopEvent(result=db)

draw_all_possible_flows(
    StatefulFlow,
    filename="basic_workflow.html",
    # Optional, can limit long event names in your workflow
    # Can help with readability
    # max_label_length=10,
)

async def main():
    w = StatefulFlow(timeout=10, verbose=False)
    result = await w.run(query="Some query")
    print(result)

asyncio.run(main())
