"""
finetune.py — Fine-tune MDLM on member data (single-GPU, plain PyTorch).

Run_3: one instance per dataset, 1× L4 GPU, no distributed training needed.
  - Accepts --dataset; reads data/{DATASET}/members.pt, writes models/{DATASET}/
  - AdamW: lr=1e-4, wd=0.01, bs=8, 5 epochs, bf16 on CUDA
  - Logs per-step + per-epoch loss to wandb: run3-{DATASET}-finetune

Outputs:
  models/{DATASET}/base_checkpoint/      — untouched pretrained weights
  models/{DATASET}/finetuned_checkpoint/ — fine-tuned weights
"""

import argparse
import math
import os
import random
import sys

import torch
import wandb
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
import config as cfg

TIME_EPSILON = 1e-3


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def mdlm_loss(model, input_ids, attention_mask, mask_token_id, device):
    """MDLM ELBO loss. Linear scheduler: p_mask = t ~ Uniform(ε, 1)."""
    B, L = input_ids.shape
    t = torch.empty(B, device=device).uniform_(TIME_EPSILON, 1.0)
    p_mask = t.unsqueeze(1).expand(B, L)

    rand = torch.rand(B, L, device=device)
    mask_positions = (rand < p_mask) & (attention_mask == 1)

    if mask_positions.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    noised = input_ids.clone()
    noised[mask_positions] = mask_token_id

    logits = model(input_ids=noised, attention_mask=attention_mask).logits
    logits_flat  = logits[mask_positions]
    targets_flat = input_ids[mask_positions]
    return torch.nn.functional.cross_entropy(logits_flat, targets_flat, reduction="mean")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    required=True, choices=cfg.ALL_DATASETS)
    parser.add_argument("--n_epochs",   type=int,   default=cfg.FT_EPOCHS)
    parser.add_argument("--batch_size", type=int,   default=cfg.FT_BATCH_SIZE)
    parser.add_argument("--lr",         type=float, default=cfg.FT_LR)
    parser.add_argument("--wd",         type=float, default=cfg.FT_WD)
    parser.add_argument("--data_base",  default="data")
    parser.add_argument("--model_base", default="models")
    parser.add_argument("--max_steps",  type=int, default=None,
                        help="Stop after N gradient steps (smoke test only)")
    args = parser.parse_args()

    data_dir  = os.path.join(args.data_base,  args.dataset)
    model_dir = os.path.join(args.model_base, args.dataset)
    os.makedirs(model_dir, exist_ok=True)

    # ----------------------------------------------------------------
    # wandb
    # ----------------------------------------------------------------
    wandb.init(
        project="da5001-mia",
        name=f"run3-{args.dataset}-finetune",
        config={
            "dataset":    args.dataset,
            "n_epochs":   args.n_epochs,
            "batch_size": args.batch_size,
            "lr":         args.lr,
            "wd":         args.wd,
        },
    )

    device = select_device()
    dtype  = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"Device: {device}, dtype: {dtype}")

    # ----------------------------------------------------------------
    # Load base model
    # ----------------------------------------------------------------
    model_path = os.environ.get("MODEL_PATH", cfg.MODEL_PATH)
    print(f"Loading base model from {model_path}")
    model = AutoModelForMaskedLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=dtype
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    mask_token_id = tokenizer.mask_token_id
    assert mask_token_id is not None, "tokenizer has no mask_token_id"
    print(f"mask_token_id = {mask_token_id}")

    # Save base checkpoint (before any training)
    base_path = os.path.join(model_dir, "base_checkpoint")
    os.makedirs(base_path, exist_ok=True)
    model.save_pretrained(base_path)
    tokenizer.save_pretrained(base_path)
    print(f"Saved base checkpoint to {base_path}")

    # ----------------------------------------------------------------
    # Load training data
    # ----------------------------------------------------------------
    members        = torch.load(os.path.join(data_dir, "members.pt"), weights_only=False)
    input_ids      = members["input_ids"]       # [N, L]
    attention_mask = members["attention_mask"]  # [N, L]
    N = input_ids.shape[0]
    print(f"Training on {N} member samples, {args.n_epochs} epochs, bs={args.batch_size}")

    # ----------------------------------------------------------------
    # Optimizer + gradient checkpointing to save VRAM
    # ----------------------------------------------------------------
    from torch.optim import AdamW
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd,
                      betas=(0.9, 0.999), eps=1e-8)

    # ----------------------------------------------------------------
    # Training loop
    # ----------------------------------------------------------------
    model.train()
    global_step = 0
    done = False

    for epoch in range(args.n_epochs):
        if done:
            break
        indices = list(range(N))
        random.shuffle(indices)
        n_batches = math.ceil(N / args.batch_size)
        epoch_loss = 0.0

        pbar = tqdm(range(n_batches), desc=f"Epoch {epoch+1}/{args.n_epochs}")
        for step in pbar:
            batch_idx = indices[step * args.batch_size : (step + 1) * args.batch_size]
            ids  = input_ids[batch_idx].to(device)
            mask = attention_mask[batch_idx].to(device)

            optimizer.zero_grad()
            loss = mdlm_loss(model, ids, mask, mask_token_id, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            loss_val     = loss.item()
            epoch_loss  += loss_val
            global_step += 1
            pbar.set_postfix(loss=f"{loss_val:.4f}")
            wandb.log({"train/loss": loss_val, "step": global_step})

            if args.max_steps and global_step >= args.max_steps:
                print(f"[SMOKE] Reached max_steps={args.max_steps}, stopping early.")
                done = True
                break

        mean_loss = epoch_loss / max(1, n_batches)
        print(f"Epoch {epoch+1} mean loss: {mean_loss:.4f}")
        wandb.log({"train/epoch_loss": mean_loss, "epoch": epoch + 1})

    # ----------------------------------------------------------------
    # Save fine-tuned checkpoint
    # ----------------------------------------------------------------
    model.eval()
    ft_path = os.path.join(model_dir, "finetuned_checkpoint")
    os.makedirs(ft_path, exist_ok=True)
    model.save_pretrained(ft_path)
    tokenizer.save_pretrained(ft_path)
    print(f"Saved finetuned checkpoint to {ft_path}")

    wandb.finish()
    print("Done.")


if __name__ == "__main__":
    main()
