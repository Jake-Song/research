from textarena_env import TextArenaEnv, TextArenaAction

textarena_url = "https://Jakemu-wordle.hf.space" # Duplicate the Space and update this!
# local_url = "http://localhost:8000"

class WordleEnv:
    def __init__(self):
        self.client = TextArenaEnv(base_url=textarena_url).sync()

    def reset(self, **kwargs) -> None | str:
        result = self.client.reset()
        # The game returns cumulative feedback each turn (new text appended at the end), so
        # we store the previous full response and slice out only the newly appended part.
        self._last_full_feedback = result.observation.messages[0].content
        self.reward = 0.0
        self.done = False
        return self._last_full_feedback

    def guess(self, guess: str) -> str:
        """
        Make a guess in the Wordle environment.

        Args:
            guess: The guessed word, formatted as '[abcde]'

        Returns:
            The feedback message from the environment.
        """
        if self.done:
            raise ValueError("Game over.")
        result = self.client.step(TextArenaAction(message=guess))
        _full_feedback = result.observation.messages[0].content
        # Just take the new feedback since the last guess
        feedback = _full_feedback[len(self._last_full_feedback):]
        self._last_full_feedback = _full_feedback
        # Penalize invalid moves
        if "You attempted an invalid move" in feedback:
            self.reward = 0.0
        else:
            self.reward = result.reward
        self.done = result.done
        return feedback

env = WordleEnv()
result = env.reset()
print(result)

guesses = ["Frame", "ghjslo", "Crane", "Sloth", "Picky", "Bound", "Grown"]
for guess in guesses:
    feedback = env.guess(guess)
    print(f"Guess: {guess}, Feedback: {feedback}, Reward: {env.reward}, Done: {env.done}")
