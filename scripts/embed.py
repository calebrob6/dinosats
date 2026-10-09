"""Embed a GeoTIFF with dino.txt patch tokens, reduce them with PCA and write a uint8 COG plus the PCA model.

Adapted from calebrob6/interactive-semantic-segmentation scripts/embed.py, with these changes:
- dino.txt (DINOv3 ViT-L/16 LVD-1689M web backbone + dino.txt vision head) with ImageNet normalisation, instead of
  the SAT-493M backbone, because the text encoder is only aligned with the LVD-1689M backbone.
- Overlapping windows blended with a Hann weight instead of centre-cropped padding. This removes the seams between
  windows (raw text similarity is window dependent) and leaves no unwritten border.
- The image is reflect-padded to a multiple of 16 px so the token grid lines up exactly with the input pixel grid.
- PCA is fit on the stitched features of the whole image rather than on separate random patches.
- The PCA model is written as JSON so the same projection can be applied to the text embedding in the browser.

Text matching against the output (all dino.txt patch matching uses the second half of the text embedding):
    t = L2-normalise(text_embeds[1024:2048])
    z = vmin + q / 255 * (vmax - vmin)          # dequantised PCA coordinates of a pixel, q = uint8 band values
    cos(t, x) ≈ t · mean + (components @ t) · z
"""

import argparse
import copy
import json
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import rasterio
import rasterio.shutil
import torch
import torch.nn.functional as F
from rasterio.io import MemoryFile
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from dinotxt import IMAGENET_MEAN, IMAGENET_STD, load_dinotxt  # noqa: E402

TOKEN_STRIDE = 16
FEATURE_DIM = 1024
TEMPLATES = ["{}", "a satellite image of {}", "an aerial photo of {}", "an overhead view of {}"]
CHECK_BACKGROUND = ["buildings", "road", "trees", "grass", "water", "parking lot", "bare soil", "houses", "concrete",
                    "sand"]
CHECK_QUERIES = ["airport runway", "ship", "golf course", "freeway interchange", "railroad tracks", "baseball stadium"]


def positions(size: int, win: int, stride: int) -> list[int]:
    p = list(range(0, size - win + 1, stride))
    return p if p[-1] == size - win else p + [size - win]


def window_features(model, img: torch.Tensor, batch, args, device: torch.device) -> torch.Tensor:
    """L2-normalised (B, t, t, 1024) patch tokens for a batch of (y, x) windows, as float16 on device."""
    win, r = args.patch_size, args.resize_factor
    t = win * r // TOKEN_STRIDE
    with torch.inference_mode():
        mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
        xb = torch.stack([img[:, y : y + win, x : x + win] for y, x in batch]).to(device).float().div_(255)
        if r != 1:
            xb = F.interpolate(xb, scale_factor=r, mode="bilinear", align_corners=False)
        xb = (xb - mean) / std
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            _, head_patches, backbone_patches = model.visual_model.get_class_and_patch_tokens(xb)
        feats = backbone_patches if args.features == "backbone" else head_patches
        return F.normalize(feats.float(), dim=-1).view(len(batch), t, t, FEATURE_DIM).half()


@torch.no_grad()
def embed(models: list, image: np.ndarray, args, devices: list[torch.device]) -> torch.Tensor:
    """image: (3, H, W) uint8 with H, W multiples of 16. Returns (h, w, 1024) L2-normalised float16 features on the CPU.

    Windows are processed in row order and batches are spread over all devices. Blending happens in a GPU buffer two
    window-heights tall; token rows that no later window touches are normalised and moved to the CPU, so GPU memory
    does not grow with the image height.
    """
    device = devices[0]
    _, H, W = image.shape
    r, win = args.resize_factor, args.patch_size
    t = win * r // TOKEN_STRIDE
    h, w = H * r // TOKEN_STRIDE, W * r // TOKEN_STRIDE
    hann = torch.sin(torch.pi * (torch.arange(t, device=device) + 0.5) / t) ** 2
    weight = (hann[:, None] * hann[None, :])[..., None]
    rows = 2 * t
    acc = torch.zeros(rows, w, FEATURE_DIM, device=device)
    wsum = torch.zeros(rows, w, 1, device=device)
    out = torch.empty(h, w, FEATURE_DIM, dtype=torch.float16)
    base = 0  # token row stored in acc[0]

    def flush(upto: int):
        nonlocal base
        n = upto - base
        if n <= 0:
            return
        out[base:upto] = F.normalize(acc[:n] / wsum[:n], dim=-1).half().cpu()
        acc[: rows - n] = acc[n:].clone()
        wsum[: rows - n] = wsum[n:].clone()
        acc[rows - n :] = 0
        wsum[rows - n :] = 0
        base = upto

    img = torch.from_numpy(image)
    coords = [(y, x) for y in positions(H, win, args.stride) for x in positions(W, win, args.stride)]
    batches = [coords[i : i + args.batch_size] for i in range(0, len(coords), args.batch_size)]
    print(f"{len(coords)} windows of {win} px, stride {args.stride} px -> token grid {h} x {w} on "
          + ", ".join(map(str, devices)))
    with ExitStack() as stack:
        pools = [stack.enter_context(ThreadPoolExecutor(1)) for _ in devices]  # one worker per device
        submit = lambda i: pools[i % len(devices)].submit(  # noqa: E731
            window_features, models[i % len(devices)], img, batches[i], args, devices[i % len(devices)])
        pending = deque(submit(i) for i in range(min(len(batches), 2 * len(devices))))
        for i in tqdm(range(len(batches)), desc="Embedding"):
            feats = pending.popleft().result().to(device)
            if i + len(pending) + 1 < len(batches):
                pending.append(submit(i + len(pending) + 1))
            for (y, x), f in zip(batches[i], feats):
                ty, tx = y * r // TOKEN_STRIDE, x * r // TOKEN_STRIDE
                flush(ty)
                acc[ty - base : ty - base + t, tx : tx + t] += f.float() * weight
                wsum[ty - base : ty - base + t, tx : tx + t] += weight
    flush(h)
    return out


@torch.no_grad()
def positional_basis(model, args, device: torch.device, k: int) -> torch.Tensor:
    """Positional subspace of the backbone tokens (INSID3, models/insid3.py `_build_positional_basis`).

    A constant image carries no content, so the variation of its patch tokens across the window is positional. Returns
    (k, FEATURE_DIM) orthonormal rows: the top-k left singular vectors of the centred, L2-normalised tokens. The input
    size matches embed(), so the grid and RoPE positions are the same as for real windows.
    """
    r, win = args.resize_factor, args.patch_size
    size = win * r
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    x = (torch.zeros(1, 3, size, size, device=device) - mean) / std
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        _, head_patches, backbone_patches = model.visual_model.get_class_and_patch_tokens(x)
    feats = backbone_patches if args.features == "backbone" else head_patches
    E = F.normalize(feats.float().reshape(-1, FEATURE_DIM), dim=-1).T.double()
    E = E - E.mean(1, keepdim=True)
    U, _, _ = torch.linalg.svd(E, full_matrices=False)
    return U[:, :k].T.float().contiguous()


def write_float32(path: Path, a: torch.Tensor, what: str) -> None:
    path.write_bytes(a.cpu().numpy().astype("<f4").tobytes())
    print(f"Wrote {path} ({path.stat().st_size / 1e6:.1f} MB, {what})")


@torch.no_grad()
def positional_inv_norm(q8: torch.Tensor, vmin, vmax, mean, comps, basis, chunk: int = 1 << 18) -> torch.Tensor:
    """1 / |(I - B^T B) x| for every pixel, where x = mean + dequantised codes @ comps is the embedding the browser
    scores against. The browser multiplies its debiased cosine by this instead of computing it at load time."""
    q8, vmin, vmax = q8.view(-1, q8.shape[-1]), vmin.to(q8.device), vmax.to(q8.device)
    mean, comps, basis = mean.to(q8.device), comps.to(q8.device), basis.to(q8.device)
    step = (vmax - vmin) / 255
    out = torch.empty(q8.shape[0], device=q8.device)
    for i in range(0, q8.shape[0], chunk):
        x = mean + (vmin + q8[i : i + chunk].float() * step) @ comps
        x = x - (x @ basis.T) @ basis
        out[i : i + chunk] = x.norm(dim=1).clamp_min(1e-3).reciprocal()
    return out


def write_posnorm(out: Path, mean, comps, vmin, vmax, basis, device: torch.device, max_tokens: int) -> None:
    """Writes <out stem>.posnorm.bin for the finest COG level with at most max_tokens tokens, which is the level the
    browser loads in full (`max_global_tokens` in config.json) and scores with the positional correction."""
    with rasterio.open(out) as src:
        overviews = src.overviews(1)
        level = 0
        for lvl, f in enumerate([1, *overviews]):
            level = lvl
            if -(-src.height // f) * -(-src.width // f) <= max_tokens:
                break
    with rasterio.open(out, overview_level=level - 1 if level else None) as src:
        q8 = torch.from_numpy(src.read()).permute(1, 2, 0).contiguous().to(device)
        h, w = src.height, src.width
    inv = positional_inv_norm(q8, vmin, vmax, mean, comps, basis)
    write_float32(out.with_suffix(".posnorm.bin"), inv, f"{h} x {w} float32, COG level {level}")


@torch.no_grad()
def fit_pca(X: torch.Tensor, k: int, n_samples: int, seed: int, device: torch.device):
    """Exact PCA via the covariance eigendecomposition. Returns mean (D,), components (k, D), explained ratio (k,)."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(X.shape[0], generator=g)[:n_samples]
    S = X[idx].to(device).double()
    mean = S.mean(0)
    S -= mean
    evals, evecs = torch.linalg.eigh(S.T @ S / (S.shape[0] - 1))
    evals, evecs = evals.flip(0), evecs.flip(1)
    comps = evecs[:, :k].T.contiguous()
    comps *= torch.sign(comps.gather(1, comps.abs().argmax(1, keepdim=True)))  # deterministic signs
    return mean.float(), comps.float(), (evals[:k] / evals.sum()).float(), len(idx)


@torch.no_grad()
def text_features(model, tokenizer, prompts: list[str], device: torch.device) -> torch.Tensor:
    out = []
    for p in prompts:
        tokens = tokenizer.tokenize([s.format(p) for s in TEMPLATES]).to(device)
        f = F.normalize(model.encode_text(tokens)[:, FEATURE_DIM:].float(), dim=-1).mean(0)
        out.append(F.normalize(f, dim=-1))
    return torch.stack(out)


@torch.no_grad()
def check_fidelity(model, tokenizer, X, mean, comps, q8, vmin, vmax, device, logit_scale):
    """Compare text scores from the full features with scores computed the browser way from the uint8 PCA output."""
    prompts = CHECK_BACKGROUND + CHECK_QUERIES
    T = text_features(model, tokenizer, prompts, device)
    full = X @ T.T
    z = vmin + q8.float() / 255 * (vmax - vmin)
    approx = T @ mean + z @ (comps @ T.T)

    def relative(S, i):
        idx = [i] + [j for j in range(len(CHECK_BACKGROUND)) if j != i]
        return (S[:, idx] * logit_scale).softmax(1)[:, 0]

    def pearson(a, b):
        a, b = a - a.mean(), b - b.mean()
        return float((a @ b) / (a.norm() * b.norm()))

    k = X.shape[0] // 100
    rows = {}
    print(f"\n{'prompt':<22}{'r raw':>8}{'r rel':>8}{'top1% overlap':>15}")
    for i, p in enumerate(prompts):
        rf, ra = relative(full, i), relative(approx, i)
        overlap = len(np.intersect1d(rf.topk(k).indices.cpu().numpy(), ra.topk(k).indices.cpu().numpy())) / k
        rows[p] = {"r_raw": pearson(full[:, i], approx[:, i]), "r_relative": pearson(rf, ra), "top1pct_overlap": overlap}
        print(f"{p:<22}{rows[p]['r_raw']:>8.4f}{rows[p]['r_relative']:>8.4f}{overlap:>15.3f}")
    return rows


def write_cog(path: str, arr: np.ndarray, crs, transform, vmin, vmax) -> None:
    count, height, width = arr.shape
    profile = dict(driver="GTiff", height=height, width=width, count=count, dtype="uint8", crs=crs, transform=transform)
    with MemoryFile() as mem:
        with mem.open(**profile) as dst:
            dst.write(arr)
            for b in range(count):
                dst.set_band_description(b + 1, f"pca_{b + 1:02d}")
                dst.update_tags(b + 1, vmin=f"{vmin[b]:.9g}", vmax=f"{vmax[b]:.9g}")
            dst.update_tags(quantization="uint8", quantization_format="embed.quant.v1")
        with mem.open() as src:
            rasterio.shutil.copy(src, path, driver="COG", compress="DEFLATE", predictor="YES", blocksize=256,
                                 overview_resampling="AVERAGE", bigtiff="IF_SAFER")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_fn", required=True, help="Input GeoTIFF; the first 3 bands are used as RGB (uint8)")
    ap.add_argument("--output_fn", required=True, help="Output uint8 PCA COG")
    ap.add_argument("--pca_fn", default=None, help="PCA model JSON (default: <output_fn stem>.pca.json)")
    ap.add_argument("--features", choices=["backbone", "head"], default="backbone",
                    help="backbone: last-layer backbone patch tokens, as in Meta's zero-shot segmentation notebook; "
                         "head: patch tokens after the 2 dino.txt vision head blocks")
    ap.add_argument("--patch_size", type=int, default=1024, help="Window size in input pixels")
    ap.add_argument("--stride", type=int, default=256,
                    help="Window stride in input pixels (overlap is Hann-blended). Patch features depend on their "
                         "position in the window, so a coarse stride leaves a periodic pattern on flat content such as "
                         "water; 256 (75%% overlap) removes it, 512 is 4x faster")
    ap.add_argument("--resize_factor", type=int, default=1, choices=[1, 2, 4],
                    help="Upsample each window before the forward pass; output stride becomes 16 / resize_factor px")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--n_pca_components", type=int, default=64)
    ap.add_argument("--n_pca_samples", type=int, default=1_000_000, help="Max tokens used to fit the PCA")
    ap.add_argument("--pca_quantile", type=float, default=0.999,
                    help="Central quantile of each PCA component mapped to 0..255 (1.0 = absolute min/max)")
    ap.add_argument("--device", default="cuda:0", help="Device, or a comma-separated list to spread windows over GPUs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--posbasis_k", type=int, default=500,
                    help="Components of the positional subspace written to <output stem>.posbasis.bin, plus the "
                         "per-token norm of the browser's global level in <output stem>.posnorm.bin (0 disables)")
    ap.add_argument("--posnorm_max_tokens", type=int, default=2_500_000,
                    help="Token limit of the global level; keep equal to max_global_tokens in config.json")
    ap.add_argument("--basis_only", action="store_true",
                    help="Only (re)write the positional basis and norm for an existing --output_fn and its PCA json, "
                         "without embedding --input_fn")
    ap.add_argument("--no_check", action="store_true", help="Skip the text-score fidelity check")
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_fn)
    pca_fn = Path(args.pca_fn) if args.pca_fn else out.with_suffix(".pca.json")
    if args.basis_only:
        device = torch.device(args.device.split(",")[0] if torch.cuda.is_available() else "cpu")
        model, _ = load_dinotxt()
        basis = positional_basis(model.to(device).eval(), args, device, args.posbasis_k)
        write_float32(out.with_suffix(".posbasis.bin"), basis, f"{basis.shape[0]} x {basis.shape[1]} float32")
        pca = json.loads(pca_fn.read_text())
        qz, t = pca["quantization"], lambda v: torch.tensor(v, device=device)  # noqa: E731
        write_posnorm(out, t(pca["mean"]), t(pca["components"]), t(qz["vmin"]), t(qz["vmax"]), basis, device,
                      args.posnorm_max_tokens)
        return
    for p in (out, pca_fn):
        if p.exists() and not args.overwrite:
            raise FileExistsError(f"{p} exists, use --overwrite")
    if args.patch_size % TOKEN_STRIDE or args.stride % TOKEN_STRIDE or not 0 < args.stride <= args.patch_size:
        raise ValueError("--patch_size and --stride must be multiples of 16 with 0 < stride <= patch_size")

    devices = [torch.device(d if torch.cuda.is_available() else "cpu") for d in args.device.split(",")]
    device = devices[0]
    torch.backends.cuda.matmul.allow_tf32 = True

    with rasterio.open(args.input_fn) as src:
        image = src.read([1, 2, 3])
        crs, transform, dtype = src.crs, src.transform, src.dtypes[0]
    if dtype != "uint8":
        raise ValueError(f"Expected uint8 RGB imagery, got {dtype}")
    _, H, W = image.shape
    pad_h = max(-(-H // TOKEN_STRIDE) * TOKEN_STRIDE, args.patch_size) - H
    pad_w = max(-(-W // TOKEN_STRIDE) * TOKEN_STRIDE, args.patch_size) - W
    if pad_h or pad_w:
        image = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
    print(f"Input {H} x {W}, padded to {image.shape[1]} x {image.shape[2]}; features: {args.features}")

    model, tokenizer = load_dinotxt()
    model = model.eval()
    logit_scale = model.logit_scale.exp().item()
    models = [copy.deepcopy(model).to(d) for d in devices[1:]]
    model = model.to(device)
    models.insert(0, model)

    tic = time.time()
    feats = embed(models, image, args, devices)
    del models[1:], image
    h, w, _ = feats.shape
    X = feats.view(-1, FEATURE_DIM)
    print(f"Embedded in {time.time() - tic:.1f}s")

    tic = time.time()
    mean, comps, evr, n_fit = fit_pca(X, args.n_pca_components, args.n_pca_samples, args.seed, device)
    cum = evr.cumsum(0)
    print(f"PCA fit on {n_fit} tokens in {time.time() - tic:.1f}s; explained variance "
          + ", ".join(f"top-{k}: {cum[k - 1] * 100:.1f}%" for k in (8, 16, 32, 64) if k <= len(cum)))

    g = torch.Generator().manual_seed(args.seed + 1)
    sample = torch.randperm(X.shape[0], generator=g)[: min(X.shape[0], 2_000_000)]
    project = lambda x: (x.to(device).float() - mean) @ comps.T  # noqa: E731
    lo = (1 - args.pca_quantile) / 2
    qs = torch.quantile(project(X[sample[:200_000]]), torch.tensor([lo, 1 - lo], device=device), dim=0)
    vmin, vmax = qs[0], torch.maximum(qs[1], qs[0] + 1e-6)
    q8 = torch.empty(X.shape[0], args.n_pca_components, dtype=torch.uint8)
    for i in range(0, X.shape[0], 1 << 20):
        z = project(X[i : i + (1 << 20)])
        q8[i : i + (1 << 20)] = ((z - vmin) / (vmax - vmin) * 255).round_().clamp_(0, 255).to(torch.uint8).cpu()

    fidelity = None
    if not args.no_check:
        sample = sample.sort().values
        fidelity = check_fidelity(model, tokenizer, X[sample].to(device).float(), mean, comps, q8[sample].to(device),
                                  vmin, vmax, device, logit_scale)

    out_transform = transform * rasterio.Affine.scale(TOKEN_STRIDE / args.resize_factor)
    arr = q8.view(h, w, -1).permute(2, 0, 1).contiguous().numpy()
    vmin_l, vmax_l = vmin.cpu().tolist(), vmax.cpu().tolist()
    write_cog(str(out), arr, crs, out_transform, vmin_l, vmax_l)
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {arr.shape[0]} bands, {h} x {w})")
    if args.posbasis_k > 0:
        basis = positional_basis(model, args, device, args.posbasis_k)
        write_float32(out.with_suffix(".posbasis.bin"), basis, f"{basis.shape[0]} x {basis.shape[1]} float32")
        write_posnorm(out, mean, comps, vmin, vmax, basis, device, args.posnorm_max_tokens)

    r7 = lambda a: [float(f"{v:.7g}") for v in a]  # noqa: E731
    pca = {
        "format": "dinotxt.pca.v1",
        "model": "dino.txt dinov3_vitl16_dinotxt_tet1280d20h24l on DINOv3 ViT-L/16 LVD-1689M",
        "features": args.features,
        "input": Path(args.input_fn).name,
        "output": out.name,
        "window_px": args.patch_size,
        "stride_px": args.stride,
        "resize_factor": args.resize_factor,
        "token_size_px": TOKEN_STRIDE / args.resize_factor,
        "input_dim": FEATURE_DIM,
        "n_components": args.n_pca_components,
        "feature_preprocessing": "patch tokens L2-normalised per window, Hann-blended across windows, L2-normalised",
        "explained_variance_ratio": r7(evr.cpu().tolist()),
        "mean": r7(mean.cpu().tolist()),
        "components": [r7(c) for c in comps.cpu().tolist()],
        "quantization": {"dtype": "uint8", "quantile": args.pca_quantile, "vmin": r7(vmin_l), "vmax": r7(vmax_l),
                         "dequantize": "z = vmin + q / 255 * (vmax - vmin)"},
        "text": {
            "embedding_slice": [FEATURE_DIM, 2 * FEATURE_DIM],
            "templates": TEMPLATES,
            "logit_scale": logit_scale,
            "score": "t = normalize(text_embeds[1024:2048]); cos(t, x) ~= dot(t, mean) + dot(components @ t, z)",
        },
        "fidelity": fidelity,
        "fidelity_sample_tokens": None if fidelity is None else len(sample),
    }
    pca_fn.write_text(json.dumps(pca))
    print(f"Wrote {pca_fn} ({pca_fn.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
