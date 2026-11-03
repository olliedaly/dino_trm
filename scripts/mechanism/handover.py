"""Does the output stop depending on slot attention's slots before slot attention collapses
(the handover)? And does the training gradient push slot attention towards collapse?

For each run and saved epoch, on cached backbone features (first 512 val images for the
reliance test; first 256 train images, no augmentation, for the gradients):

  reliance   each image's slot-attention slots (x, which also seed y) are swapped for the
             next image's (roll by one in the batch). r = (err_rolled - err) /
             (err_other - err): 0 = the output ignores x, 1 = it follows x completely
             (err_other = the other image's normal output scored against this image).
  gradients  per batch of 8, the gradient of the run's own training loss (its step
             weights): mean squared norm and coherence (|sum g|^2 - sum |g|^2) /
             ((n - 1) sum |g|^2), about the mean pairwise cosine of the batch gradients
             (0 = noise, 1 = the same every batch), for slot attention ("sa"), the input
             projection ("proj"), the module ("module") and the decoder. Also |dL/dx| per
             batch. The projection feeds both slot attention and the module's
             cross-attention; a second pass with the module reading detached patches gives
             its gradient through slot attention alone ("proj_viasa"); the rest is through
             the module.
  push       whether each batch's update moves slot attention towards collapse, to first
             order: conc = mean over images of sum_k m_k^2, m_k = slot k's share of slot
             attention's own map (1/K = even, 1 = one slot takes everything), on the first
             128 val images. push_{grp} = -(g_b / rms(g)) . grad(conc) for the Adam-like
             step of that group (rms per parameter over the batches): sa, proj, and proj
             split by path (proj_viasa, proj_viamod; they sum to proj). Positive = the
             update makes slot attention's slots more unequal.

Writes c/handover{tag}.npz: {run}|e{e}|err, err_roll (N, T) per-image per-step errors;
{run}|e{e}|err_other (N,); {run}|e{e}|g_{group}_{sq,coh} and conc scalars;
{run}|e{e}|gx, push_{sa,proj,proj_viasa,proj_viamod} (batches,).
Caps GPU memory so a concurrent training run keeps its headroom.
HANDOVER_FAST=1 skips the gradients (reliance and conc only).
Usage: handover.py C run1,run2 epochs [tag]   (missing checkpoints are skipped; voc:<run> =
VOC-trained checkpoints/<run>, still scored on COCO images)
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "scripts")
from extract_per_image import set_seed  # noqa: E402
from dino_trm.data.coco import build_coco_loader  # noqa: E402
from dino_trm.models.full_model import DinoSlotModel  # noqa: E402

torch.cuda.set_per_process_memory_fraction(0.2)
C = sys.argv[1]
RUNS = sys.argv[2].split(",")
EPOCHS = [int(e) for e in sys.argv[3].split(",")]
TAG = sys.argv[4] if len(sys.argv) > 4 else ""
NV, NT, NC, B, BG = 512, 256, 128, 16, 8
GROUPS = {"sa": ("slot_attention.",), "proj": ("input_proj.",), "module": ("reasoner.",), "decoder": ("decoder.",)}


def ckpt(name, e):
    return f"checkpoints/{name[4:]}_epoch{e}.pt" if name.startswith("voc:") else f"checkpoints/coco/{name}_epoch{e}.pt"


def cache(split, n):
    loader = build_coco_loader(split=split, batch_size=B, num_workers=4, shuffle=False,
                               augment=False, return_masks=False)
    c = torch.load(ckpt(RUNS[0], 0), map_location="cpu", weights_only=False)
    bb = DinoSlotModel(mode=c["cfg"]["mode"], **c["cfg"]["model"]).backbone.cuda().eval()
    out = []
    with torch.no_grad():
        for batch in loader:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out.append(bb(batch["pixel_values"].cuda()).cpu())
            if sum(len(o) for o in out) >= n:
                break
    del bb
    torch.cuda.empty_cache()
    return torch.cat(out)[:n]


def target(model, f):
    t = f.float()
    if model.target_norm:
        t = (t - t.mean(1, keepdim=True)) / (t.std(1, keepdim=True) + 1e-6)
    return t


def run(model, f, roll=False, slots0=None, detach_patches=False):
    """The model's forward from backbone features, optionally with x rolled over the batch
    or with the module reading detached patches."""
    t = target(model, f)
    proj = model.input_proj(t)
    if slots0 is None:
        slots0, _ = model.slot_attention(proj)
    x = slots0.roll(1, 0) if roll else slots0
    pa = proj.detach() if detach_patches else proj
    ys = [model.reasoner(x, patches=pa)] if model.mode == "mlp_block" else model.reasoner(x, patches=pa)["y"]
    return t, slots0, [model.decoder(y)[0] for y in ys]


def err(rec, t):
    return (rec.float() - t).pow(2).mean(dim=(1, 2))


FV, FT = cache("val", NV), cache("train", NT)
print("cached", tuple(FV.shape), tuple(FT.shape), flush=True)
res = {}
for name in RUNS:
    for e in EPOCHS:
        path = ckpt(name, e)
        if not os.path.exists(path):
            print("skip", path, flush=True)
            continue
        c = torch.load(path, map_location="cuda", weights_only=False)
        model = DinoSlotModel(mode=c["cfg"]["mode"], **c["cfg"]["model"]).cuda()
        model.load_state_dict(c["model"])
        model.backbone = torch.nn.Identity()  # inputs are the cached backbone features
        k = f"{name}|e{e}"

        model.eval()
        set_seed(0)
        E, ER, EO = [], [], []
        with torch.no_grad():
            for i in range(0, NV, B):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    t, s0, rec = run(model, FV[i:i + B].cuda())
                    _, _, rec_r = run(model, FV[i:i + B].cuda(), roll=True, slots0=s0)
                E.append(torch.stack([err(r, t) for r in rec], 1).cpu())
                ER.append(torch.stack([err(r, t) for r in rec_r], 1).cpu())
                EO.append(err(rec[-1].roll(1, 0), t).cpu())
        res[f"{k}|err"], res[f"{k}|err_roll"], res[f"{k}|err_other"] = (
            torch.cat(a).numpy() for a in (E, ER, EO))

        params = {g: [p for n, p in model.named_parameters() if n.startswith(pre)] for g, pre in GROUPS.items()}
        # Gradient of slot attention's slot concentration (its own map) on NC val images.
        model.zero_grad(set_to_none=True)
        set_seed(0)
        conc = 0.0
        for i in range(0, NC, B):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, attn = model.slot_attention(model.input_proj(target(model, FV[i:i + B].cuda())))
            cb = attn.float().mean(-1).pow(2).sum(-1).sum() / NC
            cb.backward()
            conc += float(cb.detach())
        cps = {g: [p for p in params[g] if p.grad is not None] for g in ("sa", "proj")}
        gconc = {g: torch.cat([p.grad.float().flatten() for p in ps]) for g, ps in cps.items()}
        res[f"{k}|conc"] = conc

        if os.environ.get("HANDOVER_FAST"):
            m, mr, mo = (res[f"{k}|{s}"] for s in ("err", "err_roll", "err_other"))
            r = (mr[:, -1].mean() - m[:, -1].mean()) / (mo.mean() - m[:, -1].mean())
            print(f"{k}: err {1000 * m[:, -1].mean():.1f} reliance {r:.2f} conc {conc:.3f}", flush=True)
            del model
            torch.cuda.empty_cache()
            np.savez(f"{C}/handover{TAG}.npz", **res)
            continue
        model.train()
        lc = c["cfg"].get("loss", {})
        T = len(rec)
        w = [lc.get("intermediate_weight", 0.25)] * (T - 1) + [lc.get("final_weight", 1.0)]
        S = {g: None for g in GROUPS}
        Q = {g: 0.0 for g in GROUPS}
        G = {g: [] for g in ("sa", "proj", "proj_viasa")}
        gx = []
        nb = NT // BG
        for b in range(nb):
            model.zero_grad(set_to_none=True)
            set_seed(1000 + b)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                t, s0, rec = run(model, FT[b * BG:(b + 1) * BG].cuda())
            s0.retain_grad()
            loss = sum(wt * F.mse_loss(r.float(), t) for wt, r in zip(w, rec))
            loss.backward()
            gx.append(float(s0.grad.float().norm()))
            for g, ps in params.items():
                v = torch.cat([p.grad.float().flatten() for p in ps if p.grad is not None])
                S[g] = v if S[g] is None else S[g] + v
                Q[g] += float(v.pow(2).sum())
            for g in ("sa", "proj"):
                G[g].append(torch.cat([p.grad.float().flatten() for p in cps[g]]))
            # Second pass: the module reads detached patches -> projection gradient via SA only.
            model.zero_grad(set_to_none=True)
            set_seed(1000 + b)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                t, s0, rec = run(model, FT[b * BG:(b + 1) * BG].cuda(), detach_patches=True)
            sum(wt * F.mse_loss(r.float(), t) for wt, r in zip(w, rec)).backward()
            G["proj_viasa"].append(torch.cat([p.grad.float().flatten() for p in cps["proj"]]))
        for g in GROUPS:
            res[f"{k}|g_{g}_sq"] = Q[g] / nb
            res[f"{k}|g_{g}_coh"] = (float(S[g].pow(2).sum()) - Q[g]) / ((nb - 1) * Q[g])
        res[f"{k}|gx"] = np.array(gx)
        G = {g: torch.stack(v) for g, v in G.items()}
        G["proj_viamod"] = G["proj"] - G["proj_viasa"]
        for g in ("proj_viasa", "proj_viamod"):
            res[f"{k}|g_{g}_sq"] = float(G[g].pow(2).sum(1).mean())
            sg, qg = G[g].sum(0), float(G[g].pow(2).sum())
            res[f"{k}|g_{g}_coh"] = (float(sg.pow(2).sum()) - qg) / ((nb - 1) * qg) if qg > 0 else np.nan
        rms = {g: G[g].pow(2).mean(0).sqrt() + 1e-12 for g in ("sa", "proj")}
        for g, base in (("sa", "sa"), ("proj", "proj"), ("proj_viasa", "proj"), ("proj_viamod", "proj")):
            res[f"{k}|push_{g}"] = (-((G[g] / rms[base]) @ gconc[base])).cpu().numpy()

        m, mr, mo = (res[f"{k}|{s}"] for s in ("err", "err_roll", "err_other"))
        r = (mr[:, -1].mean() - m[:, -1].mean()) / (mo.mean() - m[:, -1].mean())
        def tstat(a):
            return f"{a.mean():+.2e} (t {a.mean() / (a.std(ddof=1) / np.sqrt(len(a))):+.1f})"
        print(f"{k}: err {1000 * m[:, -1].mean():.1f} reliance {r:.2f} conc {conc:.3f} |dL/dx| {np.mean(gx):.2e}\n"
              "   rms/coh " + "  ".join(f"{g} {np.sqrt(res[f'{k}|g_{g}_sq']):.2e}/{res[f'{k}|g_{g}_coh']:.3f}"
                                     for g in ("sa", "proj", "proj_viasa", "proj_viamod", "module", "decoder")) +
              "\n   push " + "  ".join(f"{g} {tstat(res[f'{k}|push_{g}'])}"
                                      for g in ("sa", "proj", "proj_viasa", "proj_viamod")), flush=True)
        del model, S, G
        torch.cuda.empty_cache()
        np.savez(f"{C}/handover{TAG}.npz", **res)  # after every checkpoint, so a cut-off run keeps its results
print("handover done", flush=True)
