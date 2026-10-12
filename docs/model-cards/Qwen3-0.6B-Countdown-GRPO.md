---
base_model: Qwen/Qwen3-0.6B
library_name: peft
license: apache-2.0
language:
  - en
pipeline_tag: text-generation
tags:
  - reinforcement-learning
  - grpo
  - lora
  - reasoning
  - countdown
  - crucible-rl
model-index:
  - name: Qwen3-0.6B-Countdown-GRPO
    results:
      - task: {type: text-generation, name: Countdown arithmetic (easy, held-out)}
        dataset: {type: countdown-easy, name: Countdown 3 numbers 1-10 (256 held-out tasks)}
        metrics: [{type: accuracy, value: 97.7, name: solved (greedy, 512 tokens)}]
      - task: {type: text-generation, name: Countdown arithmetic (hard, never trained on)}
        dataset: {type: countdown-hard, name: Countdown 4 numbers 1-25 (256 tasks)}
        metrics: [{type: accuracy, value: 81.2, name: solved (greedy, 1024 tokens)}]
      - task: {type: text-generation, name: Math word problems}
        dataset: {type: openai/gsm8k, name: GSM8K, split: test}
        metrics: [{type: accuracy, value: 54.6, name: solved (greedy, 1024 tokens)}]
---

# Qwen3-0.6B-Countdown-GRPO

A LoRA adapter for [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B), trained with
reinforcement learning (GRPO) on **Countdown** arithmetic puzzles: combine the given numbers with
`+ - * /` to reach a target. It was trained in about **15 minutes on one A10G** with
[crucible_rl](https://github.com/emilzmmn04/crucible_rl), a minimal, readable framework for agentic
RL on small open models.

No human-written solutions or reasoning traces were used. The model only got a reward of 1 when a
program verified its final answer, and 0 otherwise.

![Held-out accuracy, base vs. RL-trained](assets/benchmarks-light.png)

## Results

All numbers use greedy decoding with thinking mode on. Base model and adapter are evaluated with
identical prompts, budgets and graders.

| Benchmark | Trained on? | Tasks | Token budget | Base | **This adapter** |
|---|---|---|---|---|---|
| Countdown, easy (3 numbers, 1–10) | yes (held-out tasks) | 256 | 512 | 10.5% | **97.7%** |
| Countdown, hard (4 numbers, 1–25) | **no** | 256 | 1024 | 4.7% | **81.2%** |
| GSM8K (official test set) | **no** | 1319 | 1024 | 52.9% | **54.6%** |

**What RL taught the model.** The base model reasons until it runs out of tokens and rarely
commits to an answer. After training, it tries a combination, checks it, corrects itself and
answers in a few lines:

> `(10*(5+1)) = 60. Too high. … (5*(10+1)) = 5*11 = 55. Perfect!` → `<answer>(5)*(10+1)</answer>`

**It carries over to harder puzzles.** Trained only on 3-number puzzles, the adapter solves 81%
of 4-number puzzles with values up to 25, where the base model solves under 5%.

**General math is preserved, not improved.** On GSM8K the adapter is within noise of the base
model (+1.7 points on 1319 problems). Narrow RL fine-tuning often damages general ability; here it
did not. We do not claim a GSM8K improvement.

![Held-out score during training](assets/training-curve-light.png)

## Evaluation details and caveats

- **Grading.** Countdown: the expression is parsed safely (never executed), and must use each
  given number at most once and hit the target exactly. GSM8K: the final number (in an `<answer>`
  tag or `\boxed{}`) must equal the reference answer.
- **Token budgets matter.** Scores here use a 512 or 1024-token thinking budget. Published Qwen3
  results use much longer budgets, so the base GSM8K score here is not comparable to the official
  model card.
- **Prompt format.** Countdown prompts ask for the answer inside `<answer></answer>`. The adapter
  follows this format almost always (99.6% on easy Countdown).
- **Checkpoint choice.** This is the step-40 checkpoint, chosen by the held-out Countdown score.
  Training longer can make the score worse once the task is nearly solved, so training keeps the
  best checkpoint.
- **Variance.** Training sampling is not seeded. Two runs with identical settings reached 92% and
  70% by step 20–25 and both ended above 89% at their peak.

## Training

| | |
|---|---|
| Algorithm | GRPO: group-normalized advantage, one on-policy gradient step per batch |
| Adapter | LoRA r=32, alpha=64, all linear layers |
| Batch | 8 tasks × 8 attempts per step; best checkpoint at step 40 |
| Optimizer | AdamW, lr 1e-4, gradient clip 1.0 |
| Episodes | up to 2 turns: a wrong answer gets feedback and one retry |
| Rollouts | vLLM, temperature 1.0, thinking on, max 512 new tokens per turn |
| Hardware | 1× NVIDIA A10G on Modal, about 15 minutes to the best checkpoint |

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype="auto", device_map="auto")
model = PeftModel.from_pretrained(model, "crucible-rl/Qwen3-0.6B-Countdown-GRPO")

prompt = ("Using the numbers [10, 5, 1] (each at most once) and + - * /, reach 55. "
          "Reply with only the expression inside <answer></answer>.")
ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                              enable_thinking=True, return_tensors="pt").to(model.device)
out = model.generate(ids, max_new_tokens=512)
print(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True))
```

## Reproduce

```bash
git clone https://github.com/emilzmmn04/crucible_rl && cd crucible_rl
modal run modal_app.py --steps 100 --thinking --max-new-tokens 512 --vllm --eval-every 10
# evaluate a checkpoint only:
modal run modal_app.py --steps 0 --thinking --max-new-tokens 1024 --vllm \
  --env envs.gsm8k:GSM8K --eval-tasks 1319 --init-adapter /ckpt/<run>/best
```

## Limitations

- Countdown is a narrow, synthetic task. The adapter is not a general reasoning upgrade.
- English prompts in the formats above only. Other formats were not evaluated.
- A 0.6B model: expect arithmetic mistakes on long or unusual problems.

## License

Apache-2.0, same as the base model [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B).
