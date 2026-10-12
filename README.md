# crucible_rl

**Agentic RL for small open models, in about 300 lines you can read in one sitting.**

Put a small open LLM (0.6B–8B) in a multi-turn environment, reward it with code that checks the
answer, and train it with GRPO + LoRA on a single rented GPU. No config system, no registries, no
framework to learn: five files, each readable top to bottom.

> **First result:** Qwen3-0.6B went from **2.7% → 33.6%** on a held-out Countdown benchmark in
> 100 steps (about 10 minutes on one A10G). [Details and caveats below.](#results)

## Quickstart

```bash
uvx pytest -q                         # laptop, no GPU: env + rollout logic
uv tool install modal && modal setup  # one-time: cloud GPU access
modal run modal_app.py --steps 100    # trains Qwen3-0.6B + LoRA on an A10G, prints held-out evals
```

Any `main()` argument in `crucible_rl/train.py` is a flag, e.g. `--model Qwen/Qwen3-1.7B --lr 5e-5`.

## How it works

Sample 8 tasks. Let the model attempt each task 8 times (a *group*). The environment scores each
attempt; the reward is checked by code, not by a judge. An attempt's *advantage* is its reward
minus the group average, divided by the group's spread. Increase the log-probability of every
token the model wrote, weighted by that advantage. Repeat.

Tokens from prompts and environment feedback are masked out and never trained on. Trajectories
are stored token-exactly: model output is never re-tokenized, so training sees exactly what was
sampled.

```
crucible_rl/
  env.py       what an environment is     reset(task) -> prompt, step(text) -> Step
  rollout.py   play episodes              tokens + mask (1 = model wrote it, train on it)
  policy.py    the model                  many episodes, one batched GPU call
  grpo.py      the algorithm              advantage = how much better than siblings
  train.py     the loop                   rollout -> score -> update, with held-out eval
envs/
  countdown.py example environment        reach a target with + - * /, 2 turns, verified by AST
modal_app.py   run on a cloud GPU         checkpoints go to a Modal volume
```

To understand the algorithm, read `grpo.py` (25 lines), then `rollout.py` (40 lines).

## Results

**Countdown (easy), Qwen3-0.6B + LoRA r=32, lr 1e-4, 8 tasks × 8 attempts per step, 100 steps.**
The held-out benchmark is 256 fixed tasks the model never trains on, sampled at temperature 1.

| step | 0 | 25 | 50 | 75 | 100 |
|---|---|---|---|---|---|
| held-out solved | 7 (2.7%) | 85 (33.2%) | 67 (26.2%) | 86 (33.6%) | 86 (33.6%) |

How to read it:
- The gain comes almost entirely in the first 25 steps, then flattens. Most of it is the model
  learning to answer inside `<answer>` tags, which it often skipped before training. The
  arithmetic itself improves little. That is the next thing to fix.
- The eval samples answers, so a single score can move by about ±7 tasks. The dip at step 50 is noise.
- With 4 numbers in 1..25, the base model solves 0.4%, too rarely for GRPO to get any signal.
  Start with a task the model sometimes solves.

## Write an environment

Subclass `Env` and implement three methods:

```python
class MyEnv(Env):
    max_turns = 2

    @staticmethod
    def sample_task(rng): ...          # -> any task description
    async def reset(self, task): ...   # -> first prompt (str)
    async def step(self, action): ...  # -> Step(observation, reward, done)
```

Then `modal run modal_app.py --env mymodule:MyEnv`. See `envs/countdown.py`: about 60 lines,
multi-turn, with a reward that never executes model output.

## How this compares

| | Focus | Size |
|---|---|---|
| [TRL](https://github.com/huggingface/trl) | Many post-training algorithms, production use | Large library |
| [verl](https://github.com/volcengine/verl) | Scalable RL for LLMs on many GPUs | Large framework |
| [TinyZero](https://github.com/Jiayi-Pan/TinyZero) | Reproduce R1-Zero on Countdown | Built on verl |
| [open-r1](https://github.com/huggingface/open-r1) | Reproduce DeepSeek-R1 | Built on TRL |
| **crucible_rl** | **Multi-turn agent envs, learning by reading** | **~300 lines, no RL framework** |

crucible_rl does not compete on scale. It is meant to be the codebase you read to understand
agentic RL, and the fastest way to go from a new environment idea to a training run.

## Roadmap

- [x] **M0:** GRPO + LoRA loop, Countdown env, held-out eval, first result on Modal
- [ ] **M1:** greedy eval with separate format/solve scores, vLLM for faster rollouts
- [ ] **M2:** Docker-sandboxed code-fix environment with an out-of-sandbox verifier
- [ ] **M3:** `crl` CLI, trajectory replay
- [ ] **M4:** Rust rollout scheduler and sandbox pool, remote workers

## Design rules

- Each file does one thing, and a newcomer can read it top to bottom.
- Prefer a short loop over an abstraction. Add code only when a run needs it.
- Results are shown with logs and held-out numbers, not claims.
- Rust comes in only where a benchmark shows Python is the bottleneck.

## License

MIT
