"""Export the dino.txt text encoder to ONNX for onnxruntime-web and check it against PyTorch.

input_ids: int64 [batch, seq] (seq <= 77, ending at the end-of-text token; no padding needed since attention is causal
and the output is pooled at the end-of-text token). Output text_embeds: float32 [batch, 2048], not normalised.

Writes, next to --out: the model, <model>.sha256 (the app's browser cache key) and bpe_simple_vocab_16e6.txt (for the
browser tokenizer); and web/selfcheck.json, a PyTorch reference embedding the app uses to sanity check the WebGPU
backend. A 2.2 GB full-precision export is kept in weights/onnx/ to speed up re-exports.

Usage: python scripts/export_text_encoder.py [--quant q8] [--out web/models/text_encoder.onnx]
"""

import argparse
import gzip
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import TensorProto, helper, numpy_helper

from dinotxt import BPE_VOCAB, ROOT, WEIGHTS, load_dinotxt

PROMPTS = [
    "a photo of a cat", "two cats sleeping on a pink couch", "a brown bear", "a stop sign", "a person skiing",
    "a living room", "a baseball game", "a mug and a knife", "a satellite image of a city", "an airport runway",
    "a lake", "farmland", "dense informal settlement", "buildings", "a road", "trees", "water", "sand", "snow",
    "a red jacket", "a remote control", "the sky", "a television", "grass", "a dog playing fetch in the park",
    "aerial view of a stadium", "it's raining cats & dogs!!", "Zinder, Niger", "a 3-story building", "",
]
SELFCHECK_PROMPT = "an airport runway"


class TextEncoder(torch.nn.Module):
    def __init__(self, text_model):
        super().__init__()
        self.text_model = text_model

    def forward(self, input_ids):
        return self.text_model(input_ids)


def fp16_token_embedding(model: onnx.ModelProto) -> None:
    """Store the (vocab, dim) token embedding table as float16 and cast after the lookup."""
    inits = {i.name: i for i in model.graph.initializer}
    for idx, node in enumerate(model.graph.node):
        if node.op_type != "Gather" or node.input[0] not in inits:
            continue
        init = inits[node.input[0]]
        if len(init.dims) != 2 or init.dims[0] < 10000:
            continue
        w = numpy_helper.to_array(init)
        init.CopyFrom(numpy_helper.from_array(w.astype(np.float16), init.name))
        out = node.output[0]
        node.output[0] = out + "_fp16"
        model.graph.node.insert(idx + 1, helper.make_node("Cast", [out + "_fp16"], [out], to=TensorProto.FLOAT))
        return
    raise RuntimeError("token embedding Gather not found")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quant", choices=["q4", "q8", "fp32"], default="q8",
                    help="weight-only block-quantised MatMuls (q4/q8, token embedding in fp16) or full fp32")
    ap.add_argument("--out", type=Path, default=ROOT / "web" / "models" / "text_encoder.onnx")
    ap.add_argument("--selfcheck", type=Path, default=ROOT / "web" / "selfcheck.json")
    args = ap.parse_args()

    model, tok = load_dinotxt()
    enc = TextEncoder(model.text_model).eval()
    ids = [tok.tokenize(p)[0] for p in PROMPTS]
    ids = [t[: int(t.argmax()) + 1] for t in ids]
    with torch.no_grad():
        ref = np.stack([enc(t[None])[0].numpy() for t in ids])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    (args.out.parent / "bpe_simple_vocab_16e6.txt").write_bytes(gzip.decompress(BPE_VOCAB.read_bytes()))
    sot, eot = tok.encoder["<|startoftext|>"], tok.encoder["<|endoftext|>"]
    check_ids = [sot, *tok.encode(SELFCHECK_PROMPT)][:76] + [eot]
    with torch.no_grad():
        check_emb = torch.nn.functional.normalize(enc(torch.tensor([check_ids])), dim=-1)[0]
    args.selfcheck.write_text(json.dumps({"text": SELFCHECK_PROMPT, "ids": check_ids, "embedding": check_emb.tolist()}))

    work = WEIGHTS / "onnx"
    work.mkdir(parents=True, exist_ok=True)
    fp32_path = work / "text_encoder_fp32.onnx"
    if not fp32_path.exists():
        batch, seq = torch.export.Dim("batch", min=1, max=64), torch.export.Dim("seq", min=2, max=77)
        prog = torch.onnx.export(
            enc,
            (ids[0][None],),
            input_names=["input_ids"],
            output_names=["text_embeds"],
            dynamic_shapes={"input_ids": {0: batch, 1: seq}},
            dynamo=True,
            optimize=True,
        )
        prog.save(str(fp32_path), external_data=True)

    if args.quant == "fp32":
        out_model = onnx.load(str(fp32_path))
    else:
        from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

        logging.getLogger("onnxruntime.quantization.matmul_nbits_quantizer").setLevel(logging.WARNING)
        bits = {"q4": 4, "q8": 8}[args.quant]
        q = MatMulNBitsQuantizer(onnx.load(str(fp32_path)), bits=bits, block_size=32, is_symmetric=True)
        q.process()
        out_model = q.model.model
        fp16_token_embedding(out_model)

    onnx.save(out_model, str(args.out), save_as_external_data=args.quant == "fp32")
    Path(f"{args.out}.sha256").write_text(f"{sha256(args.out)}  {args.out.name}\n")
    print(f"saved {args.out} ({args.out.stat().st_size / 1e6:.0f} MB) and {args.out.name}.sha256")

    sess = ort.InferenceSession(str(args.out), providers=["CPUExecutionProvider"])
    got = np.stack([sess.run(None, {"input_ids": t[None].numpy()})[0][0] for t in ids])
    n = lambda a: a / np.linalg.norm(a, axis=-1, keepdims=True)  # noqa: E731
    cos = (n(got) * n(ref)).sum(-1)
    print(f"cosine(onnx, torch) over {len(PROMPTS)} prompts: min {cos.min():.5f} mean {cos.mean():.5f}")


if __name__ == "__main__":
    main()
