"""Select a sparse training subset of images (deterministic stride sampling)."""

from __future__ import annotations

import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--num", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-list", default="sparse_train.txt")
    args = ap.parse_args()

    names = sorted(p.name for p in Path(args.images).glob("*")
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if args.num >= len(names):
        raise SystemExit(f"requested {args.num} of {len(names)} views")
    step = len(names) / args.num
    picked = [names[int(i * step)] for i in range(args.num)]
    with open(args.out_list, "w", encoding="utf-8") as f:
        f.write("\n".join(picked))
    print(f"wrote {len(picked)} views to {args.out_list}")


if __name__ == "__main__":
    main()
