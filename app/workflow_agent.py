from llama_index.core.workflow.resource import Resource
from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Event,
)
import asyncio
from llama_index.utils.workflow import draw_all_possible_flows
from llama_index.core.memory import Memory
from typing import Union
from llama_index.core.llms import ChatMessage
from typing import Annotated
import random

def get_memory(*args, **kwargs) -> Memory:
    return Memory.from_defaults("user_id_123", token_limit=60000)

resource = Annotated[Memory, Resource(get_memory)]

RANDOM_MESSAGES = [
    "Hello World!",
    "Python is awesome!",
    "Resources are great!",
]

class CustomStartEvent(StartEvent):
    message: str


class SecondEvent(Event):
    message: str


class ThirdEvent(Event):
    message: str

class WorkflowWithMemory(Workflow):
    @step
    async def first_step(
        self,
        ev: CustomStartEvent,
        memory: Annotated[Memory, Resource(get_memory)],
    ) -> SecondEvent:
        await memory.aput(
            ChatMessage.from_str(
                role="user", content="First step: " + ev.message
            )
        )
        return SecondEvent(message=RANDOM_MESSAGES[random.randint(0, 2)])

    @step
    async def second_step(
        self, ev: SecondEvent, memory: Annotated[Memory, Resource(get_memory)]
    ) -> Union[ThirdEvent, StopEvent]:
        await memory.aput(
            ChatMessage(role="assistant", content="Second step: " + ev.message)
        )
        if random.randint(0, 1) == 0:
            return ThirdEvent(message=RANDOM_MESSAGES[random.randint(0, 2)])
        else:
            messages = await memory.aget_all()
            return StopEvent(result=messages)

    @step
    async def third_step(
        self, ev: ThirdEvent, memory: Annotated[Memory, Resource(get_memory)]
    ) -> StopEvent:
        await memory.aput(
            ChatMessage(role="user", content="Third step: " + ev.message)
        )
        messages = await memory.aget_all()
        return StopEvent(result=messages)
    
wf = WorkflowWithMemory(disable_validation=True)

    
async def main():
    messages = await wf.run(
        start_event=CustomStartEvent(message="Happy birthday!")
    )
    for m in messages:
        print(m.blocks[0].text)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

