"""Run episodes and record them token-exactly.

A trajectory is one flat token list plus a mask: 1 = the model generated it (we train
on it), 0 = prompt or environment output (we don't). We never re-tokenize text the
model produced, so training sees exactly the tokens that were sampled.
"""
import asyncio
from dataclasses import dataclass, field


@dataclass
class Trajectory:
    tokens: list[int] = field(default_factory=list)
    mask: list[int] = field(default_factory=list)
    reward: float = 0.0

    def add(self, ids: list[int], trained: bool):
        self.tokens += ids
        self.mask += [int(trained)] * len(ids)


async def rollout(env, task, policy) -> Trajectory:
    traj = Trajectory()
    traj.add(policy.encode(await env.reset(task), first=True), trained=False)
    for _ in range(env.max_turns):
        action = await policy.generate(traj.tokens)
        traj.add(action, trained=True)
        step = await env.step(policy.decode(action))
        if step.done:
            break
        traj.add(policy.encode(step.observation), trained=False)
    traj.reward = step.reward if step.done else 0.0
    return traj


async def rollout_groups(make_env, tasks, policy, group_size) -> list[Trajectory]:
    """GRPO needs several attempts at the same task. Returns them group by group."""
    jobs = [rollout(make_env(), t, policy) for t in tasks for _ in range(group_size)]
    return await asyncio.gather(*jobs)
