"""Loads dino.txt (Meta's DINOv3 ViT-L/16 LVD-1689M backbone + dino.txt vision head and text encoder) in PyTorch.

Needs Meta's dinov3 code cloned into dinov3/ and the weights in weights/ (see the README):
- weights/hf-dinov3-vitl16-lvd1689m/model.safetensors: the gated Hugging Face backbone
- weights/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth: the dino.txt head and text encoder
- weights/bpe_simple_vocab_16e6.txt.gz: the CLIP BPE vocabulary
"""

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dinov3"))

from dinov3.hub.dinotxt import dinov3_vitl16_dinotxt_tet1280d20h24l  # noqa: E402

WEIGHTS = ROOT / "weights"
HF_BACKBONE = WEIGHTS / "hf-dinov3-vitl16-lvd1689m" / "model.safetensors"
DINOTXT_HEAD = WEIGHTS / "dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth"
BPE_VOCAB = WEIGHTS / "bpe_simple_vocab_16e6.txt.gz"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def hf_backbone_to_dinov3(hf: dict[str, torch.Tensor], ref: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map Hugging Face DINOv3ViTModel weights onto the original dinov3 repo layout.

    `ref` is the target module's state dict; its buffers (rope periods, k-bias mask) are kept as initialised.
    """
    sd = {
        "cls_token": hf["embeddings.cls_token"],
        "storage_tokens": hf["embeddings.register_tokens"],
        "mask_token": hf["embeddings.mask_token"].reshape(ref["mask_token"].shape),
        "patch_embed.proj.weight": hf["embeddings.patch_embeddings.weight"],
        "patch_embed.proj.bias": hf["embeddings.patch_embeddings.bias"],
        "norm.weight": hf["norm.weight"],
        "norm.bias": hf["norm.bias"],
        "rope_embed.periods": ref["rope_embed.periods"],
    }
    n_layers = 1 + max(int(k.split(".")[1]) for k in hf if k.startswith("layer."))
    for i in range(n_layers):
        h, o = f"layer.{i}.", f"blocks.{i}."
        q, k, v = (hf[h + f"attention.{p}_proj.weight"] for p in "qkv")
        qb, vb = hf[h + "attention.q_proj.bias"], hf[h + "attention.v_proj.bias"]
        sd |= {
            o + "norm1.weight": hf[h + "norm1.weight"],
            o + "norm1.bias": hf[h + "norm1.bias"],
            o + "attn.qkv.weight": torch.cat([q, k, v]),
            o + "attn.qkv.bias": torch.cat([qb, torch.zeros_like(qb), vb]),
            o + "attn.qkv.bias_mask": ref[o + "attn.qkv.bias_mask"],
            o + "attn.proj.weight": hf[h + "attention.o_proj.weight"],
            o + "attn.proj.bias": hf[h + "attention.o_proj.bias"],
            o + "ls1.gamma": hf[h + "layer_scale1.lambda1"],
            o + "norm2.weight": hf[h + "norm2.weight"],
            o + "norm2.bias": hf[h + "norm2.bias"],
            o + "mlp.fc1.weight": hf[h + "mlp.up_proj.weight"],
            o + "mlp.fc1.bias": hf[h + "mlp.up_proj.bias"],
            o + "mlp.fc2.weight": hf[h + "mlp.down_proj.weight"],
            o + "mlp.fc2.bias": hf[h + "mlp.down_proj.bias"],
            o + "ls2.gamma": hf[h + "layer_scale2.lambda1"],
        }
    return sd


def load_dinotxt():
    """Return (model, tokenizer) for dino.txt on the official DINOv3 ViT-L/16 LVD-1689M backbone."""
    model, tokenizer = dinov3_vitl16_dinotxt_tet1280d20h24l(pretrained=False, bpe_path_or_url=str(BPE_VOCAB))
    backbone = model.visual_model.backbone
    backbone.load_state_dict(hf_backbone_to_dinov3(load_file(HF_BACKBONE), backbone.state_dict()), strict=True)

    head = torch.load(DINOTXT_HEAD, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(head, strict=False)
    assert not unexpected, unexpected
    assert all(k.startswith("visual_model.backbone.") for k in missing), [
        k for k in missing if not k.startswith("visual_model.backbone.")
    ]
    return model.eval(), tokenizer
