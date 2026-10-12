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

## Results

**Countdown (easy), Qwen3-0.6B + LoRA, 100 steps, about 10 min on one A10G.** Held-out benchmark:
256 fixed tasks the model never trains on, sampled at temperature 1.

| step | 0 | 25 | 50 | 75 | 100 |
|---|---|---|---|---|---|
| held-out solved | 7 (2.7%) | 85 (33.2%) | 67 (26.2%) | 86 (33.6%) | 86 (33.6%) |

```bash
modal run modal_app.py --steps 100   # 3 numbers in 1..10, lr 1e-4, LoRA r=32, 8 tasks x 8 attempts per step
```

How to read it:
- The gain comes almost entirely in the first 25 steps, then flattens. Most of it is the model
  learning to answer inside `<answer>` tags, which it often skipped before training. The arithmetic
  itself improves little.
- The eval samples answers, so a single score can move by about ±7 tasks; the dip at step 50 is noise.
- With 4 numbers in 1..25 the base model solves 0.4%, too rarely for GRPO to get any signal.

## Write an environment

Subclass `Env` and implement `sample_task`, `reset`, and `step`. See `envs/countdown.py`
for an example: about 60 lines, with a multi-turn conversation and a verifiable reward.

## Rules for this codebase

- Each file does one thing, and a newcomer can read it top to bottom.
- Prefer a short loop over an abstraction. Add code only when a run needs it.
- Rust comes in only where a benchmark shows Python is the bottleneck.
