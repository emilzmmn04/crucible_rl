"""The training loop: rollout -> score -> update, repeat.

    python -m crucible_rl.train --env envs.countdown:Countdown --model Qwen/Qwen3-0.6B
"""
import argparse, asyncio, importlib, inspect, random, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

from . import grpo
from .policy import Policy
from .rollout import rollout_groups


def load_env(spec: str):
    module, name = spec.split(":")
    return getattr(importlib.import_module(module), name)


def main(env="envs.countdown:Countdown", model="Qwen/Qwen3-0.6B", steps=200, tasks_per_step=8,
         group_size=8, lr=1e-4, lora_rank=32, max_new_tokens=256, seed=0, eval_every=25, eval_tasks=256,
         out="checkpoints/run"):
    Env, rng = load_env(env), random.Random(seed)
    tok = AutoTokenizer.from_pretrained(model)
    net = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16, device_map="auto")
    net = get_peft_model(net, LoraConfig(r=lora_rank, lora_alpha=2 * lora_rank, target_modules="all-linear"))
    net.gradient_checkpointing_enable()
    net.enable_input_require_grads()
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=lr)
    policy = Policy(net, tok, max_new_tokens)
    held_out = [Env.sample_task(random.Random(10_000 + i)) for i in range(eval_tasks)]  # fixed benchmark

    def evaluate(step):
        net.eval()
        trajs = asyncio.run(rollout_groups(Env, held_out, policy, 1))
        solved = sum(t.reward == 1 for t in trajs)
        print(f"eval step {step:4d}  held-out solved {solved}/{len(trajs)} ({solved / len(trajs):.1%})", flush=True)
        print("sample:", tok.decode(trajs[0].tokens)[-400:], flush=True)

    for step in range(steps):
        if step % eval_every == 0:
            evaluate(step)
        t0 = time.time()
        tasks = [Env.sample_task(rng) for _ in range(tasks_per_step)]
        net.eval()
        trajs = asyncio.run(rollout_groups(Env, tasks, policy, group_size))
        adv = grpo.advantages([t.reward for t in trajs], group_size)

        net.train()
        opt.zero_grad()
        for t, a in zip(trajs, adv):
            if a != 0 and sum(t.mask):  # zero advantage = no signal, skip the compute
                (grpo.loss(net, t.tokens, t.mask, a) / len(trajs)).backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()

        rewards = [t.reward for t in trajs]
        print(f"step {step:4d}  reward {sum(rewards) / len(rewards):.3f}  "
              f"solved {sum(r == 1 for r in rewards)}/{len(rewards)}  {time.time() - t0:.0f}s", flush=True)
    evaluate(steps)
    net.save_pretrained(out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name, arg in inspect.signature(main).parameters.items():
        p.add_argument("--" + name.replace("_", "-"), type=type(arg.default), default=arg.default)
    main(**vars(p.parse_args()))
