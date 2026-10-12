"""GRPO in two functions.

1. advantages: within a group of attempts at the same task, an attempt is good if it
   scored above the group mean. No value network needed.
2. loss: raise the log-probability of the model's tokens in proportion to the advantage.

We do exactly one gradient step per batch of fresh rollouts, so the policy that sampled
the data is the policy being updated. The PPO importance ratio is then always 1 and
clipping never triggers, so plain policy gradient is the exact same update.
"""
import torch


def advantages(rewards: list[float], group_size: int) -> list[float]:
    r = torch.tensor(rewards).view(-1, group_size)
    a = (r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-4)
    return a.flatten().tolist()


def loss(model, tokens: list[int], mask: list[int], advantage: float) -> torch.Tensor:
    ids = torch.tensor([tokens], device=model.device)
    logits = model(ids).logits[0, :-1].float()            # position i predicts token i+1
    # log p(next token); cross_entropy avoids a second vocab-sized copy (long thinking sequences OOM otherwise)
    logp = -torch.nn.functional.cross_entropy(logits, ids[0, 1:], reduction="none")
    m = torch.tensor(mask[1:], device=model.device, dtype=logp.dtype)
    return -advantage * (logp * m).sum() / m.sum()
