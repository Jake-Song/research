from tbench2_env import Tbench2Env, Tbench2Action

env = Tbench2Env(base_url="http://localhost:8000").sync()
result = env.reset(task_id="headless-terminal")
print(result.observation.instruction)

result = env.step(Tbench2Action(action_type="exec", command="ls -la"))
print(result.observation.output)

result = env.step(Tbench2Action(action_type="evaluate"))
print(result.reward, result.done)
print(result.observation)
env.close()