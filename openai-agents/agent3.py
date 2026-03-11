# Multiple agents
import asyncio
from agents import Agent, Runner, RunConfig

history_tutor_agent = Agent(
    name="History Tutor",
    handoff_description="Specialist agent for historical questions",
    instructions="You answer history questions clearly and concisely.",
)

math_tutor_agent = Agent(
    name="Math Tutor",
    handoff_description="Specialist agent for math questions",
    instructions="You explain math step by step and include worked examples.",
)

triage_agent = Agent(
    name="Triage Agent",
    instructions="Route each homework question to the right specialist.",
    handoffs=[history_tutor_agent, math_tutor_agent],
)


async def main():
    result = await Runner.run(
        triage_agent,
        "Who was the first president of the United States?",
        # run_config=RunConfig(model="gpt-4.1"), # default
        # run_config=RunConfig(model="o4-mini"),
        run_config=RunConfig(model="gpt-4o-mini"),
        # run_config=RunConfig(model="gpt-5.4"),
    )
    print(result.final_output)
    print(f"Answered by: {result.last_agent.name}")


if __name__ == "__main__":
    asyncio.run(main())
