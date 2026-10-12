"""Countdown: combine the given numbers with + - * / to hit the target.

Reward is fully verifiable: 1 if the expression is valid, uses each number at most
once, and equals the target. The model gets two turns, so it can fix a wrong answer.
"""
import ast, operator, random, re
from crucible_rl.env import Env, Step

OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def evaluate(expr: str) -> tuple[float, list[int]]:
    """Safely evaluate + - * / over integers. Returns (value, numbers used)."""
    used = []

    def ev(node):
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            return OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            used.append(node.value)
            return node.value
        raise ValueError("not allowed")

    return ev(ast.parse(expr, mode="eval").body), used


def score(expr: str, numbers: list[int], target: int) -> tuple[float, str]:
    try:
        value, used = evaluate(expr)
    except (ValueError, SyntaxError, ZeroDivisionError):
        return 0.0, "That is not a valid expression."
    pool = list(numbers)
    for n in used:
        if n not in pool:
            return 0.0, f"You used {n} more often than allowed."
        pool.remove(n)
    if abs(value - target) > 1e-9:
        return 0.0, f"Your expression equals {value:g}, not {target}."
    return 1.0, "Correct."


class Countdown(Env):
    max_turns = 2
    n_numbers, max_number = 3, 10  # difficulty: raise these once a 0.6B model learns the easy version

    @classmethod
    def sample_task(cls, rng: random.Random):
        numbers = [rng.randint(1, cls.max_number) for _ in range(cls.n_numbers)]
        a, b, c = rng.sample(numbers, 3)  # guarantee a solution exists
        return {"numbers": numbers, "target": rng.choice([a + b * c, a * b - c, (a + b) * c])}

    async def reset(self, task):
        self.task = task
        return (f"Using the numbers {task['numbers']} (each at most once) and + - * /, "
                f"reach {task['target']}. Reply with only the expression inside <answer></answer>.")

    async def step(self, action):
        answers = re.findall(r"<answer>(.*?)</answer>", action, re.S)  # the last one is final
        reward, feedback = score(answers[-1], **self.task) if answers else (0.0, "Missing <answer> tags.")
        return Step(observation=feedback + " Try again.", reward=reward, done=reward == 1.0)
