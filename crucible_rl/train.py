"""The training loop: rollout -> score -> update, repeat.

    python -m crucible_rl.train --env envs.countdown:Countdown --model Qwen/Qwen3-0.6B
"""
import argparse, asyncio, importlib, inspect, random, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel, get_peft_model

from . import grpo
from .policy import Policy, VLLMPolicy
from .rollout import rollout_groups


def load_env(spec: str):
    module, name = spec.split(":")
    return getattr(importlib.import_module(module), name)


def main(env="envs.countdown:Countdown", model="Qwen/Qwen3-0.6B", steps=200, tasks_per_step=8,
         group_size=8, lr=1e-4, lora_rank=32, max_new_tokens=256, thinking=False, vllm=False, init_adapter="", seed=0, eval_every=25, eval_tasks=256,
         out="checkpoints/run"):
    Env, rng = load_env(env), random.Random(seed)
    tok = AutoTokenizer.from_pretrained(model)
    net = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16, device_map="auto")
    if init_adapter:  # continue from (or, with steps=0, only evaluate) a saved adapter
        net = PeftModel.from_pretrained(net, init_adapter, is_trainable=True)
    else:
        net = get_peft_model(net, LoraConfig(r=lora_rank, lora_alpha=2 * lora_rank, target_modules="all-linear",
                                             task_type="CAUSAL_LM"))
    net.gradient_checkpointing_enable()
    net.enable_input_require_grads()
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=lr)
    if vllm:  # fast sampling; the HF model below is only used for the gradient
        policy = VLLMPolicy(model, tok, lora_rank, max_new_tokens=max_new_tokens, thinking=thinking)
        policy.sync(net)
    else:
        policy = Policy(net, tok, max_new_tokens, thinking=thinking)
    held_out = Env.held_out(eval_tasks)  # fixed benchmark

    def evaluate(step):
        net.eval()
        policy.greedy = True
        trajs = asyncio.run(rollout_groups(Env, held_out, policy, 1))
        policy.greedy = False
        n = len(trajs)
        solved = sum(t.reward == 1 for t in trajs)
        # format OK = the model answered in the expected tags at least once (Countdown's format)
        fmt = sum("<answer>" in tok.decode([x for x, m in zip(t.tokens, t.mask) if m]) for t in trajs)
        print(f"eval step {step:4d}  held-out solved {solved}/{n} ({solved / n:.1%})  "
              f"format ok {fmt}/{n} ({fmt / n:.1%})", flush=True)
        print("sample:", tok.decode(trajs[0].tokens)[-400:], flush=True)
        return solved

    best = -1
    for step in range(steps):
        if step % eval_every == 0 and (solved := evaluate(step)) > best:
            best = solved  # RL can overtrain, so keep the peak, not just the end
            net.save_pretrained(f"{out}/best")
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
        if vllm:
            policy.sync(net)

        rewards = [t.reward for t in trajs]
        print(f"step {step:4d}  reward {sum(rewards) / len(rewards):.3f}  "
              f"solved {sum(r == 1 for r in rewards)}/{len(rewards)}  {time.time() - t0:.0f}s", flush=True)
    if evaluate(steps) > best:
        net.save_pretrained(f"{out}/best")
    net.save_pretrained(out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name, arg in inspect.signature(main).parameters.items():
        p.add_argument("--" + name.replace("_", "-"), type=type(arg.default), default=arg.default)
    main(**vars(p.parse_args()))
