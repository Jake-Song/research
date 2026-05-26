import random
from concurrent.futures import ThreadPoolExecutor, as_completed

from openspiel_env import OpenSpielEnv, OpenSpielAction

# url = "https://jakemu-openspiel-env.hf.space"  # Duplicate the Space and update this!
url = "http://localhost:8002"

GAME_NAME = "2048"
NUM_WORKERS = 8
MAX_STEPS = 20


class OpenSpielTestEnv:
    def __init__(self, game_name: str = GAME_NAME):
        self.client = OpenSpielEnv(base_url=url).sync()
        self.game_name = game_name

    def reset(self):
        result = self.client.reset()
        self.legal_actions = result.observation.legal_actions
        self.info_state = result.observation.info_state
        self.reward = 0.0
        self.done = False
        return result.observation

    def step(self, action_id: int):
        if self.done:
            raise ValueError("Game over.")
        result = self.client.step(
            OpenSpielAction(action_id=action_id, game_name=self.game_name)
        )
        self.legal_actions = result.observation.legal_actions
        self.info_state = result.observation.info_state
        self.reward = result.reward or 0.0
        self.done = result.done
        return result.observation


def run_episode(worker_id: int):
    env = OpenSpielTestEnv()
    obs = env.reset()
    print(f"[w{worker_id}] reset. legal_actions={obs.legal_actions}")

    total_reward = 0.0
    steps = 0
    for t in range(MAX_STEPS):
        if env.done:
            break
        action_id = random.choice(env.legal_actions)
        env.step(action_id)
        total_reward += env.reward
        steps += 1
        print(
            f"[w{worker_id}] step {t}: action={action_id}, "
            f"reward={env.reward}, done={env.done}"
        )
    return worker_id, steps, total_reward, env.done


def main():
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
        futures = [pool.submit(run_episode, i) for i in range(NUM_WORKERS)]
        for fut in as_completed(futures):
            worker_id, steps, total_reward, done = fut.result()
            print(
                f"[w{worker_id}] finished: steps={steps}, "
                f"total_reward={total_reward}, done={done}"
            )


if __name__ == "__main__":
    main()
