# crucible_rl

**Agentic RL for small open models, in about 300 lines you can read in one sitting.**

Put a small open LLM (0.6B–8B) in a multi-turn environment, reward it with code that checks the
answer, and train it with GRPO + LoRA on a single rented GPU. No config system, no registries, no
framework to learn: five files, each readable top to bottom.

> **Result:** with thinking on, Qwen3-0.6B went from **10.5% → 97.7%** on a held-out Countdown
> benchmark in 40 GRPO steps (about 15 minutes on one A10G). [Details and caveats below.](#results)

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

All numbers are on the same held-out benchmark: 256 fixed easy Countdown tasks (3 numbers from 1–10)
the model never trains on, scored with greedy decoding. *Solved* = the verified answer hits the
target; *format OK* = the model finished with an `<answer>` tag.

### Thinking on, vLLM rollouts (current best)

```bash
modal run modal_app.py --steps 100 --thinking --max-new-tokens 512 --vllm --eval-every 10
```

| step | 0 | 10 | 20 | 30 | **40** | 50 | 60 | 70 | 80 | 90 | 100 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| solved % | 10.5 | 80.1 | 91.8 | 93.4 | **97.7** | 97.3 | 94.9 | 96.9 | 95.3 | 97.3 | 97.3 |
| format OK % | 39.1 | 85.5 | 95.7 | 95.7 | **99.6** | 99.2 | 98.4 | 100 | 98.0 | 98.8 | 98.0 |

Qwen3-0.6B + LoRA r=32, lr 1e-4, 8 tasks × 8 attempts per step, 512-token thinking budget,
about 10–20 s per step on one A10G.

- **What RL taught:** the base model reasons until it runs out of tokens and never answers (39%
  format OK). After training, it tries a combination, checks it, corrects itself and answers in a few
  lines, e.g. `(10*(5+1)) = 60. Too high. … (5*(10+1)) = 55. Perfect!`
- **Overtraining:** in an earlier 150-step run, the score peaked at 89.5% (step 75) and fell to 78.1%
  by step 150. Once almost every attempt is correct, most groups give no signal, and the remaining
  updates push the model off course. So training keeps the best checkpoint (`<run>/best`), not
  just the last one.
- **Run-to-run variance:** sampling is not seeded. Two runs with identical settings reached 69.5% and
  91.8% by step 20–25.
- **Limits:** this is easy Countdown. Harder variants and other math benchmarks are next.

### Thinking off (first run)

Without thinking, the model must answer in one shot. 100 steps moved it from 2.7% to 33.6%
(sampled eval), almost all of it from learning the `<answer>` format; the arithmetic barely
improved. With 4 numbers from 1–25, the base model solved 0.4%: too rarely for GRPO to get any
signal. Start with a task the model sometimes solves.

### Speed

| rollouts | seconds per step (thinking, 512 tokens) |
|---|---|
| HF `generate` | 127–289 |
| vLLM (`--vllm`) | 10–34 |

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
- [ ] **M1:** greedy eval with format/solve scores ✓, vLLM rollouts ✓, best checkpoint ✓, capability evals, HF Hub release
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
