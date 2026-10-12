"""Runs on a laptop: no GPU, no model. A scripted policy stands in for the LLM."""
import asyncio, random
from crucible_rl.rollout import rollout_groups
from envs.countdown import Countdown, score


def test_score():
    assert score("3 * 4 + 1", [1, 3, 4, 9], 13)[0] == 1.0
    assert score("3 * 3", [1, 3, 4, 9], 9)[0] == 0.0          # 3 used twice
    assert score("__import__('os')", [1], 1)[0] == 0.0         # no code execution


class ScriptedPolicy:
    """Answers wrong on turn one, right on turn two. Tokens are just characters."""
    def __init__(self, answer): self.answer, self.calls = answer, 0
    def encode(self, text, first=False): return [ord(c) for c in text]
    def decode(self, ids): return "".join(map(chr, ids))
    async def generate(self, tokens):
        self.calls += 1
        return self.encode("<answer>1+1</answer>" if self.calls % 2 else f"<answer>{self.answer}</answer>")


def test_multi_turn_rollout_is_token_exact():
    task = {"numbers": [2, 3, 4, 5], "target": 14}
    [traj] = asyncio.run(rollout_groups(Countdown, [task], ScriptedPolicy("2+3*4"), group_size=1))
    assert traj.reward == 1.0
    trained = "".join(chr(t) for t, m in zip(traj.tokens, traj.mask) if m)
    assert trained == "<answer>1+1</answer><answer>2+3*4</answer>"  # only model tokens are trained


def test_sampled_tasks_are_solvable_types():
    t = Countdown.sample_task(random.Random(0))
    assert len(t["numbers"]) == Countdown.n_numbers and isinstance(t["target"], int)


def test_last_answer_counts():
    env = Countdown()
    asyncio.run(env.reset({"numbers": [2, 3, 4], "target": 14}))
    step = asyncio.run(env.step("<think>maybe <answer>2+3</answer>?</think> <answer>2+3*4</answer>"))
    assert step.reward == 1.0
