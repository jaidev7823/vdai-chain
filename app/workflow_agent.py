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
from pydantic import BaseModel, Field, field_validator, field_serializer
from typing import Union

# This is a random object that we want to use in our state
class MyRandomObject:
    def __init__(self, name: str = "default"):
        self.name = name

# This is our state model
# NOTE: all fields must have defaults
class MyState(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    my_obj: MyRandomObject = Field(default_factory=MyRandomObject)
    some_key: str = Field(default="some_value")

    # This is optional, but can be useful if you want to control the serialization of your state!

    @field_serializer("my_obj", when_used="always")
    def serialize_my_obj(self, my_obj: MyRandomObject) -> str:
        return my_obj.name

    @field_validator("my_obj", mode="before")
    @classmethod
    def deserialize_my_obj(
        cls, v: Union[str, MyRandomObject]
    ) -> MyRandomObject:
        if isinstance(v, MyRandomObject):
            return v
        if isinstance(v, str):
            return MyRandomObject(v)

        raise ValueError(f"Invalid type for my_obj: {type(v)}")

class MyStatefulFlow(Workflow):
    @step
    async def start(self, ctx: Context[MyState], ev: StartEvent) -> StopEvent:
        # Allows for atomic state updates
        async with ctx.store.edit_state() as state:
            state.my_obj.name = "new_name"

        # Can also access fields directly if needed
        name = await ctx.store.get("my_obj.name")

        return StopEvent(result="Done!")

draw_all_possible_flows(
    MyStatefulFlow,
    filename="basic_workflow.html",
    # Optional, can limit long event names in your workflow
    # Can help with readability
    # max_label_length=10,
)

async def main():
    w = MyStatefulFlow(timeout=10, verbose=False)

    ctx = Context(w)
    result = await w.run(ctx=ctx)
    state = await ctx.store.get_state()
    print(state)

asyncio.run(main())
