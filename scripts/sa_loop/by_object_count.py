"""Arm F against the baseline, split by the number of objects in the image.

For each validation set, counts the objects in every image at the scoring resolution
(COCO: instance masks; VOC: connected regions of each class mask, as in the eval
protocol), then prints F - B on slot attention's map (both at 8 passes) and F at 8
passes against A at 3 on the decoder's masks, per group of images. Differences are per
image, averaged over the three seed pairs, with a 95% image-bootstrap interval.
Reads results/sa_loop/per_image. CPU only.

    source scripts/sa_loop/env.sh && uv run python scripts/sa_loop/by_object_count.py \
        > results/sa_loop/by_object_count.txt
"""
import warnings

import numpy as np

from dino_trm.eval_protocol import semantic_to_instance

DIR = "results/sa_loop/per_image"
SEEDS = ["", "_seed1", "_seed2"]
RUNS = {"coco": ("baseline_pm", "sa8_rand"), "voc": ("baseline_pm_m", "sa8_rand_voc")}
BINS = [(1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, 999)]


def object_counts(dataset: str) -> np.ndarray:
    """Objects per validation image, in eval_arms.py's image order."""
    if dataset == "coco":
        from dino_trm.data.coco import build_coco_loader
        loader = build_coco_loader(split="val", batch_size=16, image_size=336, num_workers=4,
                                   shuffle=False, return_masks=True)
    else:
        from dino_trm.data.pascal_voc import build_loader
        loader = build_loader(split="val", batch_size=16, image_size=336, grid_size=21, num_workers=4,
                              shuffle=False, cache_dir=None, full_mask=True)
    counts = []
    for batch in loader:
        for b in range(batch["label_full"].shape[0]):
            sem = batch["label_full"][b].numpy().copy()
            sem[sem == 255] = 0  # void -> background, as in the eval protocol
            inst = batch["inst_full"][b].numpy() if "inst_full" in batch else semantic_to_instance(sem)
            counts.append(len([i for i in np.unique(inst) if i != 0]))
    return np.array(counts)


def seed_mean(base: str, key: str) -> np.ndarray:
    """Per-image score, averaged over the three seeds."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # images with no object are NaN in every seed
        return np.nanmean(np.stack([np.load(f"{DIR}/{base}{s}.npz")[key] for s in SEEDS]), 0)


def boot(d: np.ndarray, n: int = 4000) -> tuple[float, float, float]:
    d = d[~np.isnan(d)]
    means = d[np.random.default_rng(0).integers(0, len(d), (n, len(d)))].mean(1)
    return 100 * d.mean(), 100 * np.percentile(means, 2.5), 100 * np.percentile(means, 97.5)


def main() -> None:
    for dataset, (base, method) in RUNS.items():
        n_obj = object_counts(dataset)
        print(f"\n== {dataset}: {len(n_obj)} images, median {np.median(n_obj):.0f} objects, "
              f"{100 * np.mean(n_obj <= 2):.0f}% with one or two")
        for metric in ("mbo_i", "fg_ari"):
            for readout, ref_passes, label in (("map", 8, "F at 8 - A at 8 (B)"), ("dec", 3, "F at 8 - A at 3")):
                ref = seed_mean(base, f"{readout}{ref_passes}|{metric}")
                diff = seed_mean(method, f"{readout}8|{metric}") - ref
                print(f" {metric}, {'slot attention map' if readout == 'map' else 'decoder masks'}: {label}")
                for lo, hi in BINS:
                    keep = (n_obj >= lo) & (n_obj <= hi)
                    if keep.sum() < 20:
                        continue
                    mean, low, high = boot(diff[keep])
                    name = f"{lo}" if lo == hi else (f"{lo}+" if hi == 999 else f"{lo}-{hi}")
                    print(f"   {name:>5} objects  {keep.sum():4d} images  baseline {100 * np.nanmean(ref[keep]):5.1f}  "
                          f"difference {mean:+.2f} [{low:+.2f}, {high:+.2f}]")


if __name__ == "__main__":
    main()
