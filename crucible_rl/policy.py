"""A Hugging Face model behind an async generate().

Many episodes call generate() at once; requests are collected into one batch so the
GPU does a single forward pass for all of them.
"""
import asyncio
import torch


class Policy:
    def __init__(self, model, tokenizer, max_new_tokens=256, temperature=1.0, thinking=False):
        self.model, self.tok = model, tokenizer
        self.max_new_tokens, self.temperature, self.thinking = max_new_tokens, temperature, thinking
        self.greedy = False  # eval switches this on so scores only move when the model does
        self.pending: list[tuple[list[int], asyncio.Future]] = []

    def encode(self, text: str, first=False) -> list[int]:
        msgs = [{"role": "user", "content": text}]
        ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True,
                                           enable_thinking=self.thinking, return_dict=False)
        if not first:  # continuing a conversation: start a new turn after the model's
            ids = self.tok.encode("\n", add_special_tokens=False) + ids
        return ids

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=True)

    async def generate(self, tokens: list[int]) -> list[int]:
        fut = asyncio.get_running_loop().create_future()
        self.pending.append((tokens, fut))
        if len(self.pending) == 1:
            asyncio.get_running_loop().create_task(self._flush())
        return await fut

    async def _flush(self):
        await asyncio.sleep(0)  # let every episode that is ready enqueue first
        batch, self.pending = self.pending, []
        try:
            outs = await asyncio.to_thread(self._generate_batch, [t for t, _ in batch])
        except Exception as e:  # fail the waiting episodes instead of hanging forever
            for _, fut in batch:
                fut.set_exception(e)
            return
        for (_, fut), out in zip(batch, outs):
            fut.set_result(out)

    def _generate_batch(self, prompts: list[list[int]], chunk=32) -> list[list[int]]:
        # cap the GPU batch so a big eval doesn't run out of memory
        return [out for i in range(0, len(prompts), chunk) for out in self._generate(prompts[i:i + chunk])]

    @torch.no_grad()
    def _generate(self, prompts: list[list[int]]) -> list[list[int]]:
        self.tok.padding_side = "left"
        enc = self.tok.pad({"input_ids": prompts}, return_tensors="pt").to(self.model.device)
        out = self.model.generate(**enc, do_sample=not self.greedy,
                                  temperature=None if self.greedy else self.temperature, top_p=None, top_k=None,
                                  max_new_tokens=self.max_new_tokens, use_cache=True, pad_token_id=self.tok.pad_token_id)
        new = out[:, enc["input_ids"].shape[1]:].tolist()
        return [cut_at_eos(seq, self.tok.eos_token_id, self.tok.pad_token_id) for seq in new]


def cut_at_eos(seq, eos, pad):
    """Keep tokens up to and including the first EOS; drop padding."""
    for i, t in enumerate(seq):
        if t == eos:
            return seq[: i + 1]
        if t == pad:
            return seq[:i]
    return seq
