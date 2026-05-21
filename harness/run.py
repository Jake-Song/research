import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


CASES_PATH = Path(__file__).with_name("cases.json")
MAX_STEPS = 8


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict[str, Any]


@dataclass(frozen=True)
class FinalAnswer:
    answer: str


Action = ToolCall | FinalAnswer
TranscriptEvent = dict[str, Any]


def lookup_note(notes: dict[str, str], key: str) -> str:
    return notes[key]


def add_integers(a: int, b: int) -> str:
    return str(a + b)


def call_tool(case: dict[str, Any], tool_call: ToolCall) -> str:
    if tool_call.name == "lookup_note":
        return lookup_note(case["notes"], tool_call.args["key"])
    if tool_call.name == "add_integers":
        return add_integers(tool_call.args["a"], tool_call.args["b"])
    raise ValueError(f"unknown tool: {tool_call.name}")


class ExampleAgent:
    def next_action(self, case: dict[str, Any], transcript: list[TranscriptEvent]) -> Action:
        case_id = case["id"]
        observations = [
            event["result"]
            for event in transcript
            if event["type"] == "tool_observation"
        ]

        if case_id == "lookup-capital":
            if not observations:
                return ToolCall("lookup_note", {"key": "france_capital"})
            return FinalAnswer(observations[-1])

        if case_id == "add-two-numbers":
            if not observations:
                return ToolCall("add_integers", {"a": 17, "b": 25})
            return FinalAnswer(observations[-1])

        if case_id == "lookup-and-add":
            if len(observations) == 0:
                return ToolCall("lookup_note", {"key": "first_word"})
            if len(observations) == 1:
                return ToolCall("lookup_note", {"key": "second_word"})
            if len(observations) == 2:
                return ToolCall(
                    "add_integers",
                    {"a": len(observations[0]), "b": len(observations[1])},
                )
            return FinalAnswer(observations[-1])

        raise ValueError(f"example agent has no policy for case: {case_id}")


def load_cases() -> list[dict[str, Any]]:
    return json.loads(CASES_PATH.read_text())


def select_cases(cases: list[dict[str, Any]], case_id: str | None) -> list[dict[str, Any]]:
    if case_id is None:
        return cases

    selected = [case for case in cases if case["id"] == case_id]
    if selected:
        return selected

    known = ", ".join(case["id"] for case in cases)
    raise SystemExit(f"unknown case {case_id!r}; known cases: {known}")


def run_case(case: dict[str, Any], agent: ExampleAgent) -> dict[str, Any]:
    transcript: list[TranscriptEvent] = [
        {"type": "user_prompt", "prompt": case["prompt"]},
    ]
    tool_names: list[str] = []
    final_answer = ""
    error = ""

    for _ in range(MAX_STEPS):
        try:
            action = agent.next_action(case, transcript)
        except Exception as exc:
            error = str(exc)
            break

        if isinstance(action, FinalAnswer):
            final_answer = action.answer
            transcript.append({"type": "final_answer", "answer": final_answer})
            break

        tool_names.append(action.name)
        transcript.append(
            {"type": "tool_call", "name": action.name, "args": action.args}
        )

        try:
            result = call_tool(case, action)
        except Exception as exc:
            error = str(exc)
            transcript.append(
                {
                    "type": "tool_error",
                    "name": action.name,
                    "error": error,
                }
            )
            break

        transcript.append(
            {
                "type": "tool_observation",
                "name": action.name,
                "result": result,
            }
        )
    else:
        error = f"agent exceeded max steps: {MAX_STEPS}"

    required_tools = set(case["required_tools"])
    used_tools = set(tool_names)
    passed = (
        error == ""
        and final_answer == case["expected_answer"]
        and required_tools.issubset(used_tools)
    )

    return {
        "id": case["id"],
        "passed": passed,
        "final_answer": final_answer,
        "expected_answer": case["expected_answer"],
        "missing_tools": sorted(required_tools - used_tools),
        "error": error,
        "transcript": transcript,
    }


def print_report(results: list[dict[str, Any]]) -> None:
    for result in results:
        status = "PASS" if result["passed"] else "FAIL"
        print(f"{status} {result['id']}")
        print(f"  answer:   {result['final_answer']!r}")
        print(f"  expected: {result['expected_answer']!r}")
        if result["missing_tools"]:
            print(f"  missing tools: {', '.join(result['missing_tools'])}")
        if result["error"]:
            print(f"  error: {result['error']}")
        print("  transcript:")
        for event in result["transcript"]:
            print(f"    {json.dumps(event, sort_keys=True)}")
        print()

    passed = sum(1 for result in results if result["passed"])
    print(f"{passed}/{len(results)} cases passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the agent tool harness example.")
    parser.add_argument("--case", help="Run one case by id.")
    return parser.parse_args()


def main() -> Literal[0, 1]:
    args = parse_args()
    cases = select_cases(load_cases(), args.case)
    agent = ExampleAgent()
    results = [run_case(case, agent) for case in cases]
    print_report(results)
    return 0 if all(result["passed"] for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())
