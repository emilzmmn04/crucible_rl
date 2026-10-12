"""GSM8K: grade-school math word problems with a single numeric answer.

A fixed dataset rather than generated puzzles: train on the train split, benchmark on the official
test split (1319 problems) via held_out(). The reward is 1 if the final answer equals the reference.
The final answer is the last <answer> tag, or the last \\boxed{} (Qwen's habit), so a model isn't
scored as wrong just for using standard math notation. Data comes straight from OpenAI's repo,
so there's no extra dependency.
"""
import functools, json, os, random, re, urllib.request
from crucible_rl.env import Env, Step

URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/{}.jsonl"


@functools.cache
def load(split: str) -> list[dict]:
    path = os.path.expanduser(f"~/.cache/crucible_rl/gsm8k_{split}.jsonl")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(URL.format(split), path)
    with open(path) as f:
        rows = [json.loads(line) for line in f]
    return [{"question": r["question"], "answer": number(r["answer"].split("####")[-1])} for r in rows]


def number(text: str) -> float | None:
    """'$1,234.50.' -> 1234.5; None if there is no number."""
    m = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return float(m[-1].replace(",", "").rstrip(".")) if m else None


class GSM8K(Env):
    max_turns = 1

    @staticmethod
    def sample_task(rng: random.Random):
        return rng.choice(load("train"))

    @classmethod
    def held_out(cls, n):
        return load("test")[:n]

    async def reset(self, task):
        self.task = task
        return task["question"] + "\nGive only the final number inside <answer></answer>."

    async def step(self, action):
        answers = re.findall(r"<answer>(.*?)</answer>|\\boxed\{([^{}]*)\}", action, re.S)
        answers = [a or b for a, b in answers]
        got = number(answers[-1]) if answers else None
        correct = got is not None and abs(got - self.task["answer"]) < 1e-6
        return Step(reward=float(correct), done=True)
