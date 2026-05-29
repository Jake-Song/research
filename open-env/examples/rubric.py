from openenv.core.rubrics import Rubric
from types import SimpleNamespace


class MessageLengthRubric(Rubric):
    """Reward 1.0 if the message is 5–20 characters long, else 0.0."""

    def forward(self, action, observation) -> float:
        length = len(action.message)
        return 1.0 if 5 <= length <= 20 else 0.0

rubric = MessageLengthRubric()
obs = SimpleNamespace()

# action = SimpleNamespace(message="hi")
# score = rubric(action, obs)
# print(f"'hi'           → {score}  (last_score={rubric.last_score})")

# action = SimpleNamespace(message="hello world")
# score = rubric(action, obs)
# print(f"'hello world'  → {score}  (last_score={rubric.last_score})")

# action = SimpleNamespace(message="this message is way too long for the rubric")
# score = rubric(action, obs)
# print(f"long message   → {score}  (last_score={rubric.last_score})")

def log_score(rubric, action, obs, result):
    print(f"{type(rubric).__name__}: {result:.2f}")

rubric.register_forward_hook(log_score)              # fires after forward()
rubric.register_forward_pre_hook(lambda r, a, o: None)  # fires before forward()

action = SimpleNamespace(message="hello world")
_ = rubric(action, obs)  # hook fires and prints