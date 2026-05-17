import asyncio
from echo_env import CallToolAction, EchoEnv

async def main():
    # Connect to a running Space (async context manager)
    async with EchoEnv(base_url="http://localhost:8001") as client:
        # Reset the environment
        result = await client.reset()
        print("before: ", result)
        print("before: ", result.observation)  # "Echo environment ready!"

        # Send messages
        result = await client.step(
            CallToolAction(
                tool_name="echo_message",
                arguments={"message": "Hello, World!"},
            )
        )
        print("after: ", result)
        print("after: ", result.observation.result)  # "Hello, World!"
        print("after: ", result.reward)

asyncio.run(main())