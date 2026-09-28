"""Fetch a shard subset of MOVi from the public Kubric bucket.

The full movi_c/256x256 train split is 234GB, which is 9+ hours on a ~7MB/s link and
buys little: MOVi-C draws from roughly 1,000 ShapeNet assets and 1,000 HDRI
backgrounds, so past ~3,000 scenes you resample the same library. TFDS shards into
1024 independent files, so ANY subset is a valid sample and the subset is extensible —
fetching 316 now and 300 more later is identical to fetching 616 at once.

Measured: 9.51 sequences per train shard, 24 frames per sequence.

    uv run python scripts/download_movi.py --split validation          # 5.98GB, eval
    uv run python scripts/download_movi.py --split train --shards 316  # 72GB, ~3005 scenes

Parallel streams do NOT help (6.97MB/s single vs 6MB/s across four — the link is the
cap, not the server), so this downloads sequentially and stays resumable: complete
files are skipped by size, partial ones resume with an HTTP range request.

EVAL MUST USE ONE FRAME PER SEQUENCE. Frames within a sequence are correlated, so
using several per sequence inflates the apparent sample size. Correlated frames are
fine for unsupervised TRAINING.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

BUCKET = "https://storage.googleapis.com/kubric-public/tfds"
LIST = "https://storage.googleapis.com/storage/v1/b/kubric-public/o"
SEQ_PER_TRAIN_SHARD = 9.51


def shard_list(dataset: str, res: str, split: str) -> list[tuple[str, int]]:
    """(name, size) for every shard of a split, in order."""
    import requests
    prefix = f"tfds/{dataset}/{res}/1.0.0/{dataset}-{split}.tfrecord"
    out, token = [], None
    while True:
        p = {"prefix": prefix, "fields": "items(name,size),nextPageToken"}
        if token:
            p["pageToken"] = token
        j = requests.get(LIST, params=p, timeout=60).json()
        out += [(i["name"].split("/")[-1], int(i["size"])) for i in j.get("items", [])]
        token = j.get("nextPageToken")
        if not token:
            return sorted(out)


def fetch(url: str, dest: str, size: int) -> bool:
    """Resumable single-file download. Returns True if bytes were transferred."""
    import requests
    have = os.path.getsize(dest) if os.path.exists(dest) else 0
    if have == size:
        return False
    headers = {"Range": f"bytes={have}-"} if have else {}
    with requests.get(url, headers=headers, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(dest, "ab" if have else "wb") as f:
            for chunk in r.iter_content(1 << 22):
                f.write(chunk)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="movi_c", choices=["movi_a", "movi_b", "movi_c",
                                                            "movi_d", "movi_e"])
    ap.add_argument("--res", default="256x256", choices=["128x128", "256x256"])
    ap.add_argument("--split", default="validation",
                    choices=["train", "validation", "test"])
    ap.add_argument("--shards", type=int, default=None,
                    help="how many shards (default: all of the split)")
    ap.add_argument("--out", default="data/movi")
    args = ap.parse_args()

    shards = shard_list(args.dataset, args.res, args.split)
    if not shards:
        sys.exit(f"no shards listed for {args.dataset}/{args.res}/{args.split}")
    if args.shards:
        shards = shards[:args.shards]
    total = sum(s for _, s in shards)
    d = os.path.join(args.out, args.dataset, args.res)
    os.makedirs(d, exist_ok=True)

    est = total / 7e6 / 60
    print(f"{args.dataset}/{args.res}/{args.split}: {len(shards)} shards, "
          f"{total/2**30:.2f}GB  (~{est:.0f} min at 7MB/s)")
    if args.split == "train":
        print(f"  ~{len(shards)*SEQ_PER_TRAIN_SHARD:.0f} scenes "
              f"-> ~{len(shards)*SEQ_PER_TRAIN_SHARD*24/1000:.1f}k frames at 24 fps/seq")

    t0, done = time.time(), 0
    for i, (name, size) in enumerate(shards, 1):
        dest = os.path.join(d, name)
        moved = fetch(f"{BUCKET}/{args.dataset}/{args.res}/1.0.0/{name}", dest, size)
        done += size
        el = time.time() - t0
        rate = done / max(el, 1) / 2**20
        eta = (total - done) / max(done / max(el, 1), 1) / 60
        print(f"  [{i:4d}/{len(shards)}] {name.split('.')[-1]:22s} "
              f"{'skip' if not moved else 'ok  '}  {done/2**30:6.2f}/{total/2**30:.2f}GB  "
              f"{rate:5.1f}MB/s  eta {eta:5.1f}min", flush=True)

    print(f"\ndone -> {d}  ({total/2**30:.2f}GB in {(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
