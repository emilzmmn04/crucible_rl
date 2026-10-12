# crucible_rl

**Agentic RL for small open models, in under 300 lines you can read in one sitting.**

Put a small open LLM (0.6B–8B) in a multi-turn environment, reward it with code that checks the
answer, and train it with GRPO + LoRA on a single rented GPU. No config system, no registries, no
framework to learn: five files, each readable top to bottom.

> **Result:** RL on easy Countdown took Qwen3-0.6B from **10.5% → 97.7%** on held-out puzzles in
> about 15 minutes on one A10G. It carries over to harder puzzles it never saw (**4.7% → 81.2%**),
> and GSM8K stays where it was (52.9% → 54.6%). [Recipe and details.](recipes/countdown-qwen3-0.6b)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="recipes/countdown-qwen3-0.6b/img/benchmarks-dark.svg">
  <img alt="Held-out accuracy, base vs. RL-trained" src="recipes/countdown-qwen3-0.6b/img/benchmarks-light.svg" width="720">
</picture>

## Quickstart

```bash
uvx pytest -q                         # laptop, no GPU: env + rollout logic
uv tool install modal && modal setup  # one-time: cloud GPU access
modal run modal_app.py --steps 100 --thinking --max-new-tokens 512 --vllm --eval-every 10
                                      # trains Qwen3-0.6B + LoRA on an A10G, prints held-out evals,
                                      # keeps the best checkpoint
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
  gsm8k.py     dataset environment        math word problems; official test split as benchmark
modal_app.py   run on a cloud GPU         train, evaluate, push a model to the HF Hub
recipes/       experiments                one folder per experiment: commands, results, charts, card
tools/         helpers                    plots.py renders a recipe's charts
```

To understand the algorithm, read `grpo.py` (25 lines), then `rollout.py` (40 lines).

## Recipes

Each experiment lives in its own folder under `recipes/`, with the exact commands, every number,
charts and the model card. Framework code never depends on a recipe.

| Recipe | Model | Result | Adapter |
|---|---|---|---|
| [countdown-qwen3-0.6b](recipes/countdown-qwen3-0.6b) | Qwen3-0.6B | Countdown 10.5% → 97.7%; hard Countdown 4.7% → 81.2% (never trained on); GSM8K preserved | [crucible-rl/Qwen3-0.6B-Countdown-GRPO](https://huggingface.co/crucible-rl/Qwen3-0.6B-Countdown-GRPO) |

## Write an environment

Subclass `Env` and implement three methods:

```python
class MyEnv(Env):
    max_turns = 2

    @staticmethod
    def sample_task(rng): ...          # -> any task description
    async def reset(self, task): ...   # -> first prompt (str)
    async def step(self, action): ...  # -> Step(observation, reward, done)

    @classmethod
    def held_out(cls, n): ...          # optional: a fixed benchmark, e.g. a dataset's test split
```

Then `modal run modal_app.py --env mymodule:MyEnv`. See `envs/countdown.py` (generated puzzles,
multi-turn, a reward that never executes model output) and `envs/gsm8k.py` (a fixed dataset).

## How this compares

| | Focus | Size |
|---|---|---|
| [TRL](https://github.com/huggingface/trl) | Many post-training algorithms, production use | Large library |
| [verl](https://github.com/volcengine/verl) | Scalable RL for LLMs on many GPUs | Large framework |
| [TinyZero](https://github.com/Jiayi-Pan/TinyZero) | Reproduce R1-Zero on Countdown | Built on verl |
| [open-r1](https://github.com/huggingface/open-r1) | Reproduce DeepSeek-R1 | Built on TRL |
| **crucible_rl** | **Multi-turn agent envs, learning by reading** | **<300 lines, no RL framework** |

crucible_rl does not compete on scale. It is meant to be the codebase you read to understand
agentic RL, and the fastest way to go from a new environment idea to a training run.

## Roadmap

- [x] **M0:** GRPO + LoRA loop, Countdown env, held-out eval, first result on Modal
- [x] **M1:** greedy eval with format/solve scores, vLLM rollouts, best checkpoint, GSM8K + hard Countdown evals, first model on the HF Hub
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
