# crucible_rl

Reinforcement learning for LLM agents, small enough to read in one sitting.

```
 env.py      what an environment is      reset() -> prompt, step(text) -> Step
 rollout.py  play episodes               tokens + mask (1 = model wrote it, train on it)
 policy.py   the model                   many episodes, one batched GPU call
 grpo.py     the algorithm               advantage = how much better than siblings
 train.py    the loop                    rollout -> score -> update, repeat
```

## The algorithm in one paragraph

Sample 8 tasks. Let the model attempt each task 8 times (a *group*). The environment
scores each attempt; the reward is checked by code, not by a judge. An attempt's
*advantage* is its reward minus the group average, divided by the group's spread. Increase the
log-probability of every token the model wrote, weighted by that advantage. Repeat.
Tokens from prompts and environment feedback are masked out and never trained on.

## Run

```bash
uvx pytest -q                                   # laptop, no GPU: env + rollout logic
pip install modal && modal setup
modal run modal_app.py --steps 200              # trains Qwen3-0.6B + LoRA on a cloud A10G
```

## Write an environment

Subclass `Env` and implement `sample_task`, `reset`, and `step`. See `envs/countdown.py`
for an example: about 60 lines, with a multi-turn conversation and a verifiable reward.

## Rules for this codebase

- Each file does one thing, and a newcomer can read it top to bottom.
- Prefer a short loop over an abstraction. Add code only when a run needs it.
- Rust comes in only where a benchmark shows Python is the bottleneck.
