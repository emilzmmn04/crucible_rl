"""The environment contract. An episode is reset() followed by step() until done."""
import random
from dataclasses import dataclass


@dataclass
class Step:
    observation: str = ""  # shown to the model next turn (ignored when done)
    reward: float = 0.0    # only the reward of the final step is used
    done: bool = False


class Env:
    max_turns = 1

    @staticmethod
    def sample_task(rng: random.Random):
        """Return one task. The env decides what a task looks like."""
        raise NotImplementedError

    async def reset(self, task) -> str:
        """Start an episode, return the first prompt."""
        raise NotImplementedError

    async def step(self, action: str) -> Step:
        """Apply the model's text action, return what happened."""
        raise NotImplementedError
