"""Download NAIP quarter-quad tiles (DOQQs) from the Microsoft Planetary Computer with azcopy.

Example (3x3 block around San Diego airport):
    python scripts/download_naip.py --year 2022 --state ca --out data/naip \
        3211715_sw 3211715_se 3211716_sw 3211723_nw 3211723_ne 3211724_nw 3211723_sw 3211723_se 3211724_sw
"""

import argparse
import json
import subprocess
import urllib.request
from pathlib import Path

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
SAS = "https://planetarycomputer.microsoft.com/api/sas/v1/token/naip"


def get_json(url: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def quad_bbox(quad: str) -> list[float]:
    """Lon/lat bounds of a NAIP quarter quad such as '3211723_ne' (USGS 7.5' quads numbered 1-64 from the NW)."""
    cell, quarter = quad.split("_")
    lat0, lon0, q = int(cell[:2]), -int(cell[2:5]), int(cell[5:]) - 1
    north, west = lat0 + 1 - (q // 8) * 0.125, lon0 - 1 + (q % 8) * 0.125
    north -= 0.0625 * (quarter[0] == "s")
    west += 0.0625 * (quarter[1] == "e")
    return [west, north - 0.0625, west + 0.0625, north]


def find_item(quad: str, year: int, state: str) -> dict:
    w, s, e, n = quad_bbox(quad)
    cx, cy = (w + e) / 2, (s + n) / 2
    res = get_json(f"{STAC}/search", {
        "collections": ["naip"], "bbox": [cx - 1e-4, cy - 1e-4, cx + 1e-4, cy + 1e-4],
        "datetime": f"{year}-01-01T00:00:00Z/{year}-12-31T23:59:59Z", "limit": 50,
    })
    items = [f for f in res["features"] if f"_m_{quad}_" in f["id"] and f["id"].startswith(f"{state}_")]
    if not items:
        raise LookupError(f"No {year} NAIP item for {quad}: {[f['id'] for f in res['features']]}")
    return sorted(items, key=lambda f: f["id"])[-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("quads", nargs="+", help="Quarter quads, e.g. 3211723_ne")
    ap.add_argument("--year", type=int, default=2022)
    ap.add_argument("--state", default="ca")
    ap.add_argument("--out", default="data/naip")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    items = [find_item(q, args.year, args.state) for q in args.quads]
    token = get_json(SAS)["token"]
    for item in items:
        tif, meta = out / f"{item['id']}.tif", out / f"{item['id']}.json"
        meta.write_text(json.dumps(item, indent=1))
        if tif.exists():
            print(f"{tif} exists, skipping")
            continue
        href = item["assets"]["image"]["href"]
        print(f"{item['id']}: {href}")
        subprocess.run(["azcopy", "copy", f"{href}?{token}", str(tif), "--log-level", "ERROR",
                        "--output-level", "essential"], check=True)


if __name__ == "__main__":
    main()
