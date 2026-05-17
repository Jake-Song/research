from textarena_env import TextArenaEnv

textarena_url = "https://jakemu-wordle.hf.space" # Duplicate the Space and update this!
env = TextArenaEnv(base_url=textarena_url)
result = env.reset()
print(result.observation)
