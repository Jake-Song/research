from datasets import Dataset
PROMPT = """
Create a new short 2048 strategy using only native Python code.
You are given a list of list of numbers for the current board state.
Output one action for "0", "1", "2", "3" on what is the optimal next step.
Output your new short function in backticks using the format below:
```python
def strategy(board):
    return "0" # Example
```
All helper functions should be inside def strategy. Only output the short function `strategy`.
""".strip()

dataset = Dataset.from_dict(
    {"prompt": [[{"role": "user", "content": PROMPT}] for _ in range(3000)]}
)

print(dataset)