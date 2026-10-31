"""Train a MiniGPT under one tokenization regime.

Both runs (char / subword) share the same architecture and optimizer; only
the tokenizer changes. Loss curves and sample generations are written to
`--out` so the notebook can load and compare them.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from tqdm import tqdm

from .char_tokenizer import CharTokenizer
from .model import GPTConfig, MiniGPT
from .subword_tokenizer import SubwordTokenizer
from .utils import DEFAULT_CORPUS_CSV, bits_per_char, load_corpus, make_batches, perplexity, split_corpus


def build_tokenizer(kind: str, corpus: str, vocab_size: int):
    if kind == "char":
        return CharTokenizer().train([corpus])
    if kind == "subword":
        return SubwordTokenizer().train([corpus], vocab_size=vocab_size)
    raise ValueError(f"unknown tokenizer {kind}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", choices=["char", "subword"], required=True)
    p.add_argument("--data", type=str, default=DEFAULT_CORPUS_CSV,
                   help="path to shakespeare_lines.csv")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--vocab-size", type=int, default=8000, help="only used for subword")
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--eval-interval", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-embd", type=int, default=256)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@torch.no_grad()
def estimate_loss(model, val_ids, args, batches: int = 20) -> float:
    model.eval()
    losses = []
    for _ in range(batches):
        x, y = make_batches(val_ids, args.batch_size, args.block_size, device=args.device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    text = load_corpus(args.data)
    train_text, val_text = split_corpus(text)

    tokenizer = build_tokenizer(args.tokenizer, train_text, args.vocab_size)
    tok_path = out_dir / ("tokenizer.json" if args.tokenizer == "subword" else "vocab.json")
    tokenizer.save(tok_path)

    train_ids = torch.tensor(tokenizer.encode(train_text), dtype=torch.long)
    val_ids = torch.tensor(tokenizer.encode(val_text), dtype=torch.long)
    # Per-token loss isn't comparable across tokenizers: a BPE token carries
    # several characters. Normalizing to bits per character is.
    chars_per_token = len(val_text) / val_ids.numel()

    cfg = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
    )
    model = MiniGPT(cfg).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = AdamW(model.parameters(), lr=args.lr)

    history: list[dict] = []
    t0 = time.time()
    pbar = tqdm(range(args.steps), desc=f"train[{args.tokenizer}]")
    for step in pbar:
        x, y = make_batches(train_ids, args.batch_size, args.block_size, device=args.device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % args.eval_interval == 0 or step == args.steps - 1:
            val_loss = estimate_loss(model, val_ids, args)
            history.append(
                {
                    "step": step,
                    "train_loss": loss.item(),
                    "val_loss": val_loss,
                    "val_bpc": bits_per_char(val_loss, chars_per_token),
                    "elapsed_s": time.time() - t0,
                }
            )
            pbar.set_postfix(train=f"{loss.item():.3f}", val=f"{val_loss:.3f}")

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    final = history[-1]
    summary = {
        "tokenizer": args.tokenizer,
        "vocab_size": tokenizer.vocab_size,
        "n_params": n_params,
        "chars_per_token": chars_per_token,
        "steps": args.steps,
        "device": args.device,
        "val_loss": final["val_loss"],
        "val_perplexity": perplexity(final["val_loss"]),
        "val_bpc": final["val_bpc"],
        "train_time_s": final["elapsed_s"],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, out_dir / "model.pt")

    # quick generation sample so we have something qualitative on disk
    model.eval()
    prompt_ids = torch.tensor([tokenizer.encode("\n")], dtype=torch.long, device=args.device)
    out_ids = model.generate(prompt_ids, max_new_tokens=400, temperature=0.9, top_k=40)
    sample = tokenizer.decode(out_ids[0].tolist())
    (out_dir / "sample.txt").write_text(sample, encoding="utf-8")
    print(f"\nsaved checkpoint, history, sample to {out_dir}")


if __name__ == "__main__":
    main()
