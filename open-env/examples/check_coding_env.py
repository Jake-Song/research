import asyncio
from coding_env import CodeAction, CodingEnv

async def main():
    # Create environment from Docker image
    client = await CodingEnv.from_docker_image("coding-env:latest")

    async with client:
        # Reset
        result = await client.reset()
        print(f"Reset complete: exit_code={result.observation.exit_code}")

        # Execute Python code
        code_samples = [
            "print('Hello, World!')",
            "x = 5 + 3\nprint(f'Result: {x}')",
            "import math\nprint(math.pi)"
        ]

        for code in code_samples:
            result = await client.step(CodeAction(code=code))
            print(f"Code: {code}")
            print(f"  → stdout: {result.observation.stdout.strip()}")
            print(f"  → exit_code: {result.observation.exit_code}")

asyncio.run(main())