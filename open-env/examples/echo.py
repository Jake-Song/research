import asyncio
from echo_env import CallToolAction, EchoEnv

url = "https://openenv-echo-env.hf.space"

async def main():
    # Connect to a running Space (async context manager)
    async with EchoEnv(base_url=url) as client:
        # Reset the environment
        result = await client.reset()
        print("result: ", result)
        print("obs.result: ", result.observation)  # "Echo environment ready!"
        
        tools = await client.list_tools()
        print([t.name for t in tools])

        # Send messages
        result = await client.step(
            CallToolAction(
                tool_name="echo_message",
                arguments={"message": "Hello World!"},
            )
        )
        result = await client.step(
            CallToolAction(
                tool_name="echo_with_length",
                arguments={"message": "Hello World!"},
            )
        )
        print("result: ", result)
        print("obs.result: ", result.observation.result)  # "Hello"
        print("reward: ", result.reward)
       
asyncio.run(main())