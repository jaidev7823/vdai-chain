from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Event,
)
import asyncio
from llama_index.utils.workflow import draw_all_possible_flows
from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

class Step2Event(Event):
    query: str


class Step3Event(Event):
    query: str

class Step2BEvent(Event):
    query: str

class MainWorkflow(Workflow):
    @step
    async def start(self, ev: StartEvent) -> Step2Event:
        print("Starting up")
        return Step2Event(query=ev.query)

    @step
    async def step_two(self, ev: Step2Event) -> Step3Event:
        print("Sending an email")
        return Step3Event(query=ev.query)

    @step
    async def step_three(self, ev: Step3Event) -> StopEvent:
        print("Finishing up")
        return StopEvent(result=ev.query)

class CustomWorkflow(MainWorkflow):
    @step
    async def step_two(self, ev: Step2Event) -> Step2BEvent:
        print("Sending an email")
        return Step2BEvent(query=ev.query)

    @step
    async def step_two_b(self, ev: Step2BEvent) -> Step3Event:
        print("Also sending a text message")
        return Step3Event(query=ev.query)

async def main():
    w = MainWorkflow(timeout=30, verbose=True)
    handler = w.run(query="Start the workflow.")

    final_result = await handler
    print("Final result", final_result)

    draw_all_possible_flows(MainWorkflow, filename="basic_workflow.html")


if __name__ == "__main__":
    asyncio.run(main())