"""Score a run on the OTHER dataset's validation set (train on COCO, test on VOC, or the reverse).

Uses eval_arms.py's own scoring loop unchanged; only the checkpoint loading differs, because
eval_arms.load() refuses a run whose training set is not the test set.

    source scripts/sa_loop/env.sh && uv run python scripts/sa_loop/cross_eval.py --train coco --test voc --runs baseline_pm,sa8_rand --out-dir results/sa_loop/cross
Writes {out dir}/{run}__on_{test}.npz in eval_arms.py's format.
"""
import argparse, json, os, sys
import numpy as np, torch
sys.path.insert(0, "scripts/sa_loop")
import eval_arms as E
from dino_trm.models.full_model import DinoSlotModel


def make_load(train_ds):
    def load(run, _test_ds):
        d = E.CKPT_DIRS[train_ds]
        epochs = torch.load(os.path.join(d, f"{run}_epoch0.pt"), map_location="cpu", weights_only=False)["cfg"]["optim"]["epochs"]
        path = os.path.join(d, f"{run}_epoch{epochs - 1}.pt")
        c = torch.load(path, map_location="cuda", weights_only=False)
        cfg = c["cfg"]
        assert c["epoch"] == epochs - 1 and cfg["mode"] in ("baseline", "sa_rec")
        assert cfg["data"].get("dataset", "voc") == train_ds, (run, cfg["data"].get("dataset"))
        model = DinoSlotModel(mode=cfg["mode"], **cfg["model"]).cuda()
        model.load_state_dict(c["model"])
        return model.eval(), cfg, path
    return load


ap = argparse.ArgumentParser()
ap.add_argument("--train", required=True); ap.add_argument("--test", required=True)
ap.add_argument("--runs", required=True); ap.add_argument("--out-dir", required=True)
a = ap.parse_args()
os.makedirs(a.out_dir, exist_ok=True)
E.load = make_load(a.train)
for run in a.runs.split(","):
    dst = os.path.join(a.out_dir, f"{run}__on_{a.test}.npz")
    if os.path.exists(dst):
        print(f"{run}: {dst} exists, skipped", flush=True); continue
    out = E.run_one(run, a.test, [3, 8], ["map", "dec"], 0)
    meta = out.pop("_meta"); meta.update(passes=[3, 8], readouts=["map", "dec"], eval_seeds=[0], trained_on=a.train, tested_on=a.test)
    np.savez_compressed(dst, meta=json.dumps(meta), **out)
    show = "  ".join(f"{k} {100 * np.nanmean(out[f'map8|{k}']):.2f}" for k in ("fg_ari", "mbo_i", "mbo_c", "ari_bg"))
    print(f"{run} (trained on {a.train}) on {a.test}: {len(out['err8'])} images | map at 8 passes: {show}", flush=True)
