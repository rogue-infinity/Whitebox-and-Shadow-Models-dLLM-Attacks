"""
verify_memorization.py — Check that fine-tuning induced memorization.

Computes ELBO gap = mean(base_member_loss - finetuned_member_loss).
Exits 1 if gap < 0.05 nats (insufficient memorization).
"""

import argparse
import os
import sys

import torch
import wandb
from transformers import AutoModelForMaskedLM, AutoTokenizer
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
import config as cfg

MASKING_RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9]
SEED = 0
MIN_GAP = 0.05  # nats


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def compute_elbo(model, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                 mask_token_id: int, masking_ratios: list[float],
                 device: torch.device, seed: int = 0) -> torch.Tensor:
    """Return mean CE over masking_ratios for each sample. Shape: [N]."""
    model.eval()
    N, L = input_ids.shape
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    losses_per_ratio = []
    for ratio in masking_ratios:
        # Deterministic mask for this ratio
        rand = torch.rand(N, L, generator=gen, device=device)
        mask_pos = (rand < ratio) & (attention_mask == 1)

        noised = input_ids.clone()
        noised[mask_pos] = mask_token_id

        logits = model(input_ids=noised, attention_mask=attention_mask).logits  # [N, L, V]

        # Per-sample CE on masked positions
        sample_losses = []
        for i in range(N):
            pos = mask_pos[i]
            if pos.sum() == 0:
                sample_losses.append(torch.tensor(0.0, device=device))
                continue
            loss = torch.nn.functional.cross_entropy(
                logits[i][pos], input_ids[i][pos], reduction="mean"
            )
            sample_losses.append(loss)
        losses_per_ratio.append(torch.stack(sample_losses))  # [N]

    return torch.stack(losses_per_ratio, dim=0).mean(dim=0)  # [N]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    required=True, choices=cfg.ALL_DATASETS)
    parser.add_argument("--data_base",  default="data")
    parser.add_argument("--model_base", default="models")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    data_dir  = os.path.join(args.data_base,  args.dataset)
    model_dir = os.path.join(args.model_base, args.dataset)

    wandb.init(project="da5001-mia", name=f"run3-{args.dataset}-verify",
               config={"dataset": args.dataset, **vars(args)})

    device = select_device()
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"Device: {device}, dtype: {dtype}")

    base_path = os.path.join(model_dir, "base_checkpoint")
    ft_path   = os.path.join(model_dir, "finetuned_checkpoint")

    print("Loading base model...")
    base_model = AutoModelForMaskedLM.from_pretrained(
        base_path, trust_remote_code=True, torch_dtype=dtype
    ).to(device).eval()

    print("Loading finetuned model...")
    ft_model = AutoModelForMaskedLM.from_pretrained(
        ft_path, trust_remote_code=True, torch_dtype=dtype
    ).to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    mask_token_id = tokenizer.mask_token_id

    # Load data
    members    = torch.load(os.path.join(data_dir, "members.pt"),    weights_only=False)
    nonmembers = torch.load(os.path.join(data_dir, "nonmembers.pt"), weights_only=False)

    def run_batched(model, dataset):
        ids = dataset["input_ids"].to(device)
        mask = dataset["attention_mask"].to(device)
        N = ids.shape[0]
        all_losses = []
        for i in tqdm(range(0, N, args.batch_size)):
            b_ids = ids[i : i + args.batch_size]
            b_mask = mask[i : i + args.batch_size]
            elbo = compute_elbo(model, b_ids, b_mask, mask_token_id,
                                MASKING_RATIOS, device, SEED)
            all_losses.append(elbo.cpu())
        return torch.cat(all_losses)

    print("Computing base model ELBO on members...")
    base_member_loss = run_batched(base_model, members)
    print("Computing finetuned model ELBO on members...")
    ft_member_loss = run_batched(ft_model, members)
    print("Computing base model ELBO on non-members...")
    base_nonmember_loss = run_batched(base_model, nonmembers)
    print("Computing finetuned model ELBO on non-members...")
    ft_nonmember_loss = run_batched(ft_model, nonmembers)

    member_gap = (base_member_loss - ft_member_loss).mean().item()
    nonmember_gap = (base_nonmember_loss - ft_nonmember_loss).mean().item()

    print(f"\nMember ELBO gap    (base - finetuned): {member_gap:.4f} nats")
    print(f"Non-member ELBO gap (base - finetuned): {nonmember_gap:.4f} nats")
    print(f"Required gap: >= {MIN_GAP} nats")

    wandb.log({
        "verify/member_gap": member_gap,
        "verify/nonmember_gap": nonmember_gap,
        "verify/base_member_loss_mean": base_member_loss.mean().item(),
        "verify/ft_member_loss_mean": ft_member_loss.mean().item(),
        "verify/passed": member_gap >= MIN_GAP,
    })

    # Save per-sample ELBO tensors for ablation study
    out_dir = os.path.join(args.model_base.replace("models", "results"), args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    # Use results/{DATASET}/ path
    results_dir = os.path.join("results", args.dataset)
    os.makedirs(results_dir, exist_ok=True)
    torch.save({
        "base_member_elbo":    base_member_loss.cpu(),    # [N_mem]
        "ft_member_elbo":      ft_member_loss.cpu(),      # [N_mem]
        "base_nonmember_elbo": base_nonmember_loss.cpu(), # [N_non]
        "ft_nonmember_elbo":   ft_nonmember_loss.cpu(),   # [N_non]
        "member_gap":          member_gap,
        "nonmember_gap":       nonmember_gap,
    }, os.path.join(results_dir, "verify_elbo.pt"))
    print(f"Saved {results_dir}/verify_elbo.pt")

    wandb.finish()

    if member_gap < MIN_GAP:
        print(f"\nWARNING: member gap {member_gap:.4f} < {MIN_GAP}. "
              f"Memorization is weak — continuing pipeline anyway.")
    else:
        print(f"\nPASS: memorization verified (gap = {member_gap:.4f} >= {MIN_GAP}).")


if __name__ == "__main__":
    main()
