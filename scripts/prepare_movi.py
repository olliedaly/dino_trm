"""Flatten MOVi TFRecord shards into per-frame files for random-access training.

TFRecords are sequential, which fights a shuffling DataLoader. This unpacks the shards
once into plain files. Frames are written VERBATIM — the records already hold PNG
bytes, so there is no re-encode and no lossy step.

    data/movi/frames/{split}/{seq}_{frame:02d}.png       RGB frame
    data/movi/frames/{split}/{seq}_{frame:02d}_seg.png   instance ids, 0 = background
    data/movi/frames/{split}/index.json                  {seq: [category per instance]}

Segmentation id k maps to ``categories[k-1]``; id 0 is background. Instances can be
absent from a given frame (occluded or off-screen), so the ids present are a subset.

FRAME SAMPLING. Training uses **DINOSAUR's protocol: 9 random frames per clip**, which
is how that paper turned MOVi into an image dataset (9,737 clips x 9 = the 87,633
training images it reports). Random beats a fixed stride here for a reason specific to
MOVi: objects are tossed and settle over the clip, so a stride starting at 0 puts the
same mid-air configurations into every sequence's contribution, giving the training set
a systematic temporal profile. Random sampling averages that out.

Validation takes ONE frame per sequence (default index 12, once objects have settled).
That is not fussiness — frames within a clip are correlated, so several per clip would
inflate the effective sample size of the eval.
Correlation is acceptable in training because the objective is unsupervised.

    uv run python scripts/prepare_movi.py --split validation
    uv run python scripts/prepare_movi.py --split train --frames-per-clip 9
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random

from dino_trm.data.tfrecord import parse_example, records

SRC = "data/movi/{ds}/{res}"
DST = "data/movi/frames/{split}"
VAL_FRAME = 12


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="movi_c")
    ap.add_argument("--res", default="256x256")
    ap.add_argument("--split", default="validation",
                    choices=["train", "validation", "test"])
    ap.add_argument("--frames-per-clip", type=int, default=9,
                    help="train only: random frames per clip (DINOSAUR's protocol is 9)")
    ap.add_argument("--val-frame", type=int, default=VAL_FRAME)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    src = SRC.format(ds=args.dataset, res=args.res)
    shards = sorted(glob.glob(os.path.join(src, f"{args.dataset}-{args.split}.tfrecord-*")))
    if not shards:
        raise SystemExit(f"no {args.split} shards under {src} — run scripts/download_movi.py")
    dst = DST.format(split=args.split)
    os.makedirs(dst, exist_ok=True)

    index: dict[str, list[int]] = {}
    if os.path.exists(f"{dst}/index.json"):
        index = json.load(open(f"{dst}/index.json"))

    n_seq = n_frame = 0
    for si, shard in enumerate(shards, 1):
        for raw in records(shard):
            ex = parse_example(raw)
            name = ex["metadata/video_name"][0].decode()
            vid, seg = ex["video"], ex["segmentations"]
            keep = ([args.val_frame] if args.split != "train"
                    else sorted(rng.sample(range(len(vid)),
                                           min(args.frames_per_clip, len(vid)))))
            for t in keep:
                if t >= len(vid):
                    continue
                with open(f"{dst}/{name}_{t:02d}.png", "wb") as f:
                    f.write(vid[t])
                with open(f"{dst}/{name}_{t:02d}_seg.png", "wb") as f:
                    f.write(seg[t])
                n_frame += 1
            index[name] = list(ex["instances/category"])
            n_seq += 1
        json.dump(index, open(f"{dst}/index.json", "w"))
        print(f"  [{si:4d}/{len(shards)}] {os.path.basename(shard).split('.')[-1]:22s} "
              f"sequences {n_seq:6d}  frames {n_frame:7d}", flush=True)

    print(f"\ndone -> {dst}  {n_seq} sequences, {n_frame} frames")


if __name__ == "__main__":
    main()
