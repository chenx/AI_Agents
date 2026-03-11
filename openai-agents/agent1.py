import asyncio
from agents import Agent, Runner, RunConfig

agent = Agent(
  name="History Tutor",
  instructions="You answer history questions clearly and concisely.",
)

async def main():
  result = await Runner.run(
    agent, 
    "When did the Roman Empire fall?",
    run_config=RunConfig(model="gpt-4o-mini"),
  )
  print(result.final_output)

if __name__ == "__main__":
  asyncio.run(main())

