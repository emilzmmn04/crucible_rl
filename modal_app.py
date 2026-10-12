"""Run training on a rented GPU. Your laptop only sends code and streams logs.

    modal run modal_app.py --model Qwen/Qwen3-0.6B --steps 200
"""
import modal

image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch", "transformers", "peft", "accelerate")
         .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})  # avoid fragmentation OOMs
         .add_local_python_source("crucible_rl", "envs"))
volume = modal.Volume.from_name("crucible-rl-checkpoints", create_if_missing=True)
app = modal.App("crucible-rl", image=image)


@app.function(gpu="A10G", timeout=6 * 3600, volumes={"/ckpt": volume})
def train(**kwargs):
    from crucible_rl.train import main
    main(out="/ckpt/run", **kwargs)
    volume.commit()


@app.local_entrypoint()
def run(model: str = "Qwen/Qwen3-0.6B", env: str = "envs.countdown:Countdown", steps: int = 200,
        thinking: bool = False, max_new_tokens: int = 256, eval_every: int = 25, eval_tasks: int = 256):
    train.remote(model=model, env=env, steps=steps, thinking=thinking, max_new_tokens=max_new_tokens,
                 eval_every=eval_every, eval_tasks=eval_tasks)
