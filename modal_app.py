"""Run training on a rented GPU. Your laptop only sends code and streams logs.

    modal run modal_app.py --model Qwen/Qwen3-0.6B --steps 200
"""
import modal

image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("vllm", "peft", "accelerate")  # vllm pins compatible torch + transformers
         .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",  # avoid fragmentation OOMs
               "VLLM_USE_FLASHINFER_SAMPLER": "0"})  # its sampler JIT-compiles and needs nvcc
         .add_local_python_source("crucible_rl", "envs")
         .add_local_dir("recipes", "/root/recipes"))  # model cards + charts, for push
volume = modal.Volume.from_name("crucible-rl-checkpoints", create_if_missing=True)
app = modal.App("crucible-rl", image=image)


@app.function(gpu="A10G", timeout=6 * 3600, volumes={"/ckpt": volume},
              secrets=[modal.Secret.from_name("huggingface")])
def train(name="run", **kwargs):
    from crucible_rl.train import main
    main(out=f"/ckpt/{name}", **kwargs)  # one folder per run, so runs never overwrite each other
    volume.commit()


@app.local_entrypoint()
def run(model: str = "Qwen/Qwen3-0.6B", env: str = "envs.countdown:Countdown", steps: int = 200,
        thinking: bool = False, max_new_tokens: int = 256, eval_every: int = 25, eval_tasks: int = 256,
        vllm: bool = False, name: str = "run", init_adapter: str = ""):
    train.remote(model=model, env=env, steps=steps, thinking=thinking, max_new_tokens=max_new_tokens,
                 eval_every=eval_every, eval_tasks=eval_tasks, vllm=vllm, name=name, init_adapter=init_adapter)


@app.function(volumes={"/ckpt": volume}, secrets=[modal.Secret.from_name("huggingface")])
def push(name: str, repo: str, recipe: str):
    """Upload a run's adapter with its recipe's model card and charts to the HF Hub, private by default.

        modal run modal_app.py::push --name <run>/best --repo crucible-rl/<model> --recipe <recipe>
    """
    import shutil
    from huggingface_hub import HfApi
    folder = f"/ckpt/{name}"
    shutil.copy(f"/root/recipes/{recipe}/model-card.md", f"{folder}/README.md")
    shutil.copytree(f"/root/recipes/{recipe}/img", f"{folder}/assets", dirs_exist_ok=True)
    api = HfApi()
    api.create_repo(repo, private=True, exist_ok=True)
    api.upload_folder(folder_path=folder, repo_id=repo)
    print(f"uploaded https://huggingface.co/{repo}")
