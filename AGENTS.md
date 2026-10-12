# AGENTS.md

Context for coding agents working on crucible_rl.

## Goal

**Make agentic RL for small open models easy, readable and effective.** Take any small
open model (0.6B–8B), put it in an environment where it acts over multiple turns, gets
rewards that code can verify, and measurably improves. Everything should fit in a codebase a
person can read in an afternoon.

North star: *the nanoGPT of agentic RL.* Today: the cleanest reference implementation.
Later: the fast environment runtime (sandboxes, async rollouts, verifiers) that any trainer
can plug into.

Success looks like this:
- A held-out benchmark score that rises after training, shown with logs and plots, not claims.
- A new environment goes from idea to training run in under 15 minutes.
- Someone new understands the whole algorithm by reading `grpo.py` + `rollout.py`.

Non-goals: competing with verl/TRL on scale, supporting every algorithm, or adding features nobody has asked for.

## What this is

A minimal, readable RL framework for training small open LLM agents in multi-turn
environments with verifiable rewards (GRPO + LoRA). Goals, in order:
1. The owner can understand every line. This is a learning and portfolio project.
2. Minimal lines of code. Today the whole loop is about 290 lines.
3. Grow into a fast agentic-RL environment runtime: sandboxes, async rollouts, verifiers.

## Layout

| File | Role |
|---|---|
| `crucible_rl/env.py` | `Env` contract: `sample_task(rng)`, `async reset(task) -> str`, `async step(text) -> Step` |
| `crucible_rl/rollout.py` | Runs episodes. `Trajectory` = flat `tokens` + `mask` (1 = model-generated) + `reward` |
| `crucible_rl/policy.py` | HF model with async `generate()`; concurrent requests are batched into one GPU call. `VLLMPolicy` (`--vllm`) samples with vLLM on the same GPU, about 4–5× faster; the LoRA adapter is synced after each step |
| `crucible_rl/grpo.py` | `advantages()` (group-normalized) and `loss()` (masked policy gradient) |
| `crucible_rl/train.py` | Loop: sample tasks → rollout groups → advantages → one optimizer step. CLI flags = `main()` kwargs |
| `envs/countdown.py` | Example env: arithmetic target, 2 turns, safe AST evaluation for the reward |
| `modal_app.py` | Runs `train.main` on a cloud GPU (A10G). Checkpoints go to Modal volume `crucible-rl-checkpoints` |
| `tests/test_core.py` | No-GPU tests using a scripted character-level policy |
| `index.html` | Landing page / technical note (white, black and grey, Inter, research-paper look) |

## Rules

1. **Explain before you build.** Before writing non-trivial code, state the plan in 2–5 sentences. Afterwards, explain what changed and why in plain words. The owner is learning RL.
2. **Line budget.** Core (`crucible_rl/`) stays under ~500 lines until M2. Deleting code is better than adding it. Every PR says `+N / -M lines`.
3. **No new dependencies** without asking. Current deps: torch, transformers, peft, accelerate, modal, pytest.
4. **No new files or abstractions** unless the existing file would become harder to read without them.
5. **Tests must pass** (`uvx pytest -q`) before you call something done. If you change GPU code you couldn't test, say so explicitly.
6. **Be honest about results.** Never claim training "works" without a log that shows the reward rising.
7. **Correctness before speed.** Any optimization needs a measured before/after number.
8. **Ask before** spending money (cloud runs beyond quick tests), pushing to a remote, or changing the public API in `env.py`.
9. **Commits** are small and use imperative messages, e.g. `add held-out eval`. One idea per commit.
10. **Keep this file current.** Update the Layout and Status sections whenever you change them.

## Versions and branches

- `main` holds released versions only. Each finished milestone gets a tag: M0 = `v0.1`, M1 = `v0.2`,
  M2 = `v0.3`, and so on. `v1.0` comes when every milestone on the roadmap is done.
- Work for the next milestone happens on a branch named after its version (currently `v0.2` for M1),
  and is merged into `main` and tagged when the milestone is done.

## Invariants (do not break)

- **Token-exact trajectories.** Never re-tokenize model output. Env text is appended
  with mask 0, and generated ids with mask 1.
- **On-policy, one gradient step per batch.** This is why `grpo.loss` has no ratio or
  clipping. If you add multiple updates per batch, store sampling logprobs and add PPO clipping.
- **With `--vllm`, sampling is very slightly off-policy:** vLLM and HF compute probabilities with
  slightly different numerics. This is accepted at our scale. If it ever matters, store vLLM's
  sampling logprobs and add an importance-ratio correction.
- **Only the final step's reward counts.** An episode that hits `max_turns` gets reward 0.
- **Verifiers never execute model output as code** unless sandboxed (see `evaluate` in countdown).
- `rollout.py` and `env.py` must not import torch, so the tests run on a laptop.

## Conventions

- Prefer a short, obvious loop over an abstraction. No config classes, registries, or plugins until needed.
- Every file gets a short docstring explaining *why*. Add comments only where it isn't obvious.
- Add Rust only when a benchmark shows Python is the bottleneck (planned: rollout scheduler, sandbox pool).
- When you add something, report the change in line count.

## Environment

- The owner develops on an 8 GB M1 Mac: no CUDA, no local training. GPU work runs on Modal.
- Tests: `uvx pytest -q`
- Train: `modal run modal_app.py --steps 200` (needs `uv tool install modal && modal setup`)
- Local env check without a model: write a scripted policy as in the tests.

## Status / roadmap

- **M0 (done, 2026-10-12):** Countdown + GRPO on Modal. Easy Countdown (3 numbers, 1–10), Qwen3-0.6B,
  LoRA r=32, lr=1e-4, 100 steps (~10 min A10G): held-out (256 tasks, sampled) 2.7% → 33.6%.
  Gain came by step 25 and then plateaued; mostly learning the `<answer>` format, not better math.
  Original difficulty (4 numbers, 1–25) was too hard: baseline 0.4%, no learning signal.
- **M1 (in progress, branch `v0.2`):** greedy eval with format/solve scores, `--thinking`, `--vllm`
  (4–5× faster), best-checkpoint saving, `--init-adapter`, `CountdownHard`, `push` to the HF Hub.
  Best result: thinking + vLLM, easy Countdown 10.5% → 97.7% held-out at step 40
  (`/ckpt/countdown-think-vllm-100/best` on the Modal volume). Next: capability evals (hard
  Countdown, GSM8K), then upload to `crucible-rl/Qwen3-0.6B-Countdown-GRPO`.
- M1: held-out eval script (baseline vs trained), vLLM for faster rollouts
- M2: Docker sandboxed code-fix env with an out-of-sandbox verifier
- M3: `crl` CLI, trajectory replay
- M4: Rust scheduler and sandbox pool, remote workers, microVM isolation
