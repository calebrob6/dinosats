"""Mosaic a rectangular block of NAIP DOQQs into one RGB raster cropped to the area every pixel has data for.

Writes a lossless tiled GeoTIFF (input for scripts/embed.py) and a JPEG (YCbCr) COG for the web app.

    python scripts/make_mosaic.py data/naip/*.tif --out data/naip_2022_san_diego_3x3.tif \
        --web_out web/data/naip_2022_san_diego_3x3_rgb.tif
"""

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
import rasterio.shutil
from rasterio.merge import merge
from rasterio.windows import from_bounds

sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_naip import quad_bbox  # noqa: E402

ALIGN_PX = 16  # keeps the dino.txt token grid aligned with the pixel grid


def inner_bounds(srcs: list) -> tuple[float, float, float, float]:
    """Largest axis-aligned box covered by the outer row/column of tiles, using the quarter-quad grid in the names."""
    quads = [quad_bbox(re.search(r"m_(\d{7}_[ns][ew])_", Path(s.name).name).group(1)) for s in srcs]
    west, east = min(q[0] for q in quads), max(q[2] for q in quads)
    south, north = min(q[1] for q in quads), max(q[3] for q in quads)
    pick = lambda i, v: [s.bounds for s, q in zip(srcs, quads) if abs(q[i] - v) < 1e-6]  # noqa: E731
    return (max(b.left for b in pick(0, west)), max(b.bottom for b in pick(1, south)),
            min(b.right for b in pick(2, east)), min(b.top for b in pick(3, north)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tiles", nargs="+")
    ap.add_argument("--out", required=True, help="Lossless RGB GeoTIFF")
    ap.add_argument("--web_out", help="JPEG-compressed RGB COG for the browser")
    ap.add_argument("--jpeg_quality", type=int, default=90)
    args = ap.parse_args()

    srcs = [rasterio.open(t) for t in args.tiles]
    crs, res = srcs[0].crs, srcs[0].res
    if any(s.crs != crs or s.res != res for s in srcs):
        raise ValueError("All tiles must share a CRS and resolution")
    left, bottom, right, top = inner_bounds(srcs)
    width = int((right - left) / res[0] + 1e-6) // ALIGN_PX * ALIGN_PX
    height = int((top - bottom) / res[1] + 1e-6) // ALIGN_PX * ALIGN_PX
    bounds = (left, top - height * res[1], left + width * res[0], top)
    print(f"{len(srcs)} tiles -> {width} x {height} px, bounds {bounds}")

    tic = time.time()
    mosaic, transform = merge(srcs, bounds=bounds, res=res, indexes=[1, 2, 3], nodata=0)
    assert mosaic.shape == (3, height, width), mosaic.shape
    empty = int((mosaic.max(0) == 0).sum())
    print(f"Merged in {time.time() - tic:.0f}s; {empty} all-zero pixels ({empty / width / height:.2e})")
    for s in srcs:
        w = from_bounds(*s.bounds, transform=transform)
        print(f"  {Path(s.name).name}: rows {w.row_off:.0f}..{w.row_off + w.height:.0f}, "
              f"cols {w.col_off:.0f}..{w.col_off + w.width:.0f}")

    profile = dict(driver="GTiff", width=width, height=height, count=3, dtype="uint8", crs=crs, transform=transform,
                   tiled=True, blockxsize=512, blockysize=512, compress="DEFLATE", predictor=2, bigtiff="IF_SAFER",
                   photometric="RGB")
    tic = time.time()
    with rasterio.open(args.out, "w", **profile) as dst:
        dst.write(mosaic)
    print(f"Wrote {args.out} ({Path(args.out).stat().st_size / 1e9:.2f} GB) in {time.time() - tic:.0f}s")
    del mosaic

    if args.web_out:
        tic = time.time()
        with rasterio.open(args.out) as src:
            rasterio.shutil.copy(src, args.web_out, driver="COG", compress="JPEG", quality=args.jpeg_quality,
                                 blocksize=512, overview_resampling="AVERAGE", bigtiff="IF_SAFER")
        with rasterio.open(args.web_out) as cog:
            info = cog.tags(ns="IMAGE_STRUCTURE")
            print(f"Wrote {args.web_out} ({Path(args.web_out).stat().st_size / 1e6:.0f} MB, {info}, "
                  f"overviews {cog.overviews(1)}) in {time.time() - tic:.0f}s")


if __name__ == "__main__":
    main()
