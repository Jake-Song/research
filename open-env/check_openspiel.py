import asyncio
from openspiel_env import OpenSpielEnv, OpenSpielAction

url = "http://localhost:8002"

async def main():
    # Connect to a running Space (async context manager)
    async with OpenSpielEnv(base_url=url) as client:
        # Reset the environment
        result = await client.reset()
        print("result: ", result)
        print("obs.result: ", result.observation)  # "Echo environment ready!"
        result = await client.step(
            OpenSpielAction(action_id=0, game_name="2048")
        )
        print("result: ", result)
asyncio.run(main())