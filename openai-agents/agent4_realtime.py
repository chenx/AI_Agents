import asyncio

from agents.realtime import RealtimeAgent, RealtimeRunner

agent = RealtimeAgent(
    name="Assistant",
    instructions="You are a helpful voice assistant. Keep responses short and conversational.",
)

runner = RealtimeRunner(
    starting_agent=agent,
    config={
        "model_settings": {
            "model_name": "gpt-realtime",
            "audio": {
                "input": {
                    "format": "pcm16",
                    "transcription": {"model": "gpt-4o-mini-transcribe"},
                    "turn_detection": {
                        "type": "semantic_vad",
                        "interrupt_response": True,
                    },
                },
                "output": {
                    "format": "pcm16",
                    "voice": "ash",
                },
            },
        }
    },
)

async def main() -> None:
    session = await runner.run()

    async with session:
        await session.send_message("Say hello in one short sentence.")

        async for event in session:
            if event.type == "audio":
                print("==> audio")
                # Forward or play event.audio.data.
                pass
            elif event.type == "history_added":
                print("==> history_added")
                print(event.item)
            elif event.type == "agent_end":
                print("==> agent_end")
                # One assistant turn finished.
                break
            elif event.type == "error":
                print("==> error")
                print(f"Error: {event.error}")


if __name__ == "__main__":
    asyncio.run(main())
