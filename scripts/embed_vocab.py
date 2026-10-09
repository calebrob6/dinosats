"""Embed a list of candidate text strings with dino.txt for the app's "what is this point?" lookup.

Uses the same prompt templates, embedding slice and normalisation as scripts/embed.py and the browser, so the vectors are
directly comparable with the patch embeddings. The background prompts and example queries from web/config.json are always
included, so the app can show them without downloading the text encoder. Writes:
- <out_dir>/vocab.json: {"strings": [...], "dim": 1024, "dtype": "float16"}
- <out_dir>/vocab_embeddings.bin: float16 little-endian, shape (n_strings, dim), L2-normalised

Usage: python scripts/embed_vocab.py [--vocab scripts/vocab.txt] [--out_dir web/data] [--device cuda:0]
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from dinotxt import load_dinotxt  # noqa: E402

FEATURE_DIM = 1024
TEMPLATES = ["{}", "a satellite image of {}", "an aerial photo of {}", "an overhead view of {}"]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vocab", default=str(ROOT / "scripts" / "vocab.txt"), help="One candidate string per line")
    ap.add_argument("--config", default=str(ROOT / "web" / "config.json"), help="App config whose background and examples are added")
    ap.add_argument("--out_dir", default=str(ROOT / "web" / "data"))
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    lines = Path(args.vocab).read_text().splitlines()
    if args.config and Path(args.config).exists():
        config = json.loads(Path(args.config).read_text())
        lines += config.get("background", []) + config.get("examples", [])
    strings = list(dict.fromkeys(" ".join(s.split()) for s in lines if s.strip()))
    model, tokenizer = load_dinotxt()
    model = model.to(args.device)

    out = []
    for i in range(0, len(strings), args.batch_size):
        batch = strings[i : i + args.batch_size]
        tokens = tokenizer.tokenize([t.format(s) for s in batch for t in TEMPLATES]).to(args.device)
        f = F.normalize(model.encode_text(tokens)[:, FEATURE_DIM : 2 * FEATURE_DIM].float(), dim=-1)
        f = F.normalize(f.view(len(batch), len(TEMPLATES), -1).mean(1), dim=-1)
        out.append(f.cpu())
        print(f"{min(i + args.batch_size, len(strings))}/{len(strings)}", flush=True)
    emb = torch.cat(out).contiguous()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "vocab.json").write_text(json.dumps({"strings": strings, "dim": FEATURE_DIM, "dtype": "float16"}))
    (out_dir / "vocab_embeddings.bin").write_bytes(emb.numpy().astype("<f2").tobytes())
    print(f"wrote {len(strings)} embeddings to {out_dir}")


if __name__ == "__main__":
    main()
