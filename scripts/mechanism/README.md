# Recursion-mechanism analysis

These scripts produce the numbers in `results/coco/diagnostics/recursion_mechanism.md`. They compare T=1 (`trm_t1*`) with T=8 (`trm*`), with `baseline_pm*`, `mlp_block*` and `sa_rec_1step` as references.

## Running them

- Run every script from the repo root.
- The first argument is a cache directory `C` for intermediate arrays (several GB).
- GPU scripts need `HF_HOME=<hf cache> HF_HUB_OFFLINE=1`.
- They use eval seed 0 and batch 16, and cap their GPU memory at 20% so they can run beside training.

## Pipeline, in order

1. **Extract** (GPU; all 2,000 COCO val images)

   | script | writes |
   |---|---|
   | `cache.py C runs` | `m_{run}.npy` final decoder masks; `gt.npz` instance and class maps |
   | `cache_steps.py C runs` | `m_{run}_steps.npy` masks after every step |
   | `gpu_batch.py C runs [steps]` | `rx_{run}.npz`: per-step MSE, slots and masks. `EVAL_SEED=s` adds `_e{s}`; `steps` runs the module longer at test time (`_n{n}`); `CKPT_DIR` / `RX_PREFIX` load other checkpoint folders (e.g. the VOC-trained runs) |
   | `sa.py`, `sa_iter.py` | slot attention's own maps and the module's cross-attention; slot attention iterated at test time |
   | `patch_err.py C runs` | `pe_{run}.npy` per-patch reconstruction error |
   | `voc_x.py` | the same extraction on VOC |
   | `curve.py C runs epochs` | `curve.npz`: MSE on 512 val and 512 train images, and val masks, for each epoch checkpoint (`CURVE_TAG` env renames the file) |
   | `curve_sa.py C runs epochs` | `curve_sa.npz`: slot attention's own map at each epoch checkpoint |
   | `loc_mse.py C runs` | `rx_{run}_loc{b}.npz`: T=1 with a locality penalty on its decoder logits (b = 5, 10, 20); reconstruction error and masks |

2. **Object pairs.** `pairdist.py` → `pd_trm.npy`, one row per GT object pair: image, same class, gap in px.

3. **Per-image statistics.** Each image gets FG-ARI, ARI counting background, mBO_i, mBO_c and the pixel-pair rates (whole, same-class far/near/touching, different-class, background share).

   | script | writes | covers |
   |---|---|---|
   | `mech.py` | `mech.npz` | main contrast |
   | `mech2.py` | `mech2.npz` | T=2/4, `mlp_block`, test-time steps |
   | `mech3.py` | `mech3.npz` | eval seeds 1–2 |
   | `mech_ctrl.py C runs` | `mechc_{runs}.npz` | the control runs, and the `loc_mse.py` variants |
   | `mech_loc.py` | `mechloc.npz` | hard locality penalty on T=1's masks (geo2 `loc{b}`) |
   | `geo2.py C run variants` | `g_{run}.npz` | slot geometry and the `loc{b}` / `lam{x}` variants (read by `locprior.py`) |
   | `voc_mech.py` | `voc_mech.npz` | VOC |

4. **Probes.**
   - `probe.py` → `probe.npz`: linear probes for class, position and size on each object's best-IoU slot.
   - `probe_strat.py` → `probe_oof.npz`: the same probes split by whether the object's class appears elsewhere in the image.
   - `probe_extra.py C "name=rx file:step,..."` → `probe_extra.npz`: the same probes for other models (controls, locality priors), next to T=1 and T=8 of the same seed.

5. **Tables and figures**

   | script | report section |
   |---|---|
   | `mech_an.py` | 1, 5, 6, 11 |
   | `mech2_an.py` | 3, 5: dose-response, `mlp_block`, test-time steps, per-image MSE |
   | `collapse.py` | 3: slot collapse and convergence when iterated |
   | `size_ctx.py` | 6: object-size slope vs same-class context |
   | `voc_an.py` | 7 |
   | `rec_report.py`, `it_an.py` | 8 |
   | `ro.py` | 9: slot-attention map vs decoder masks |
   | `featsim.py C [extras]` (GPU) | 1, 10: slot sharing by distance and DINO feature similarity, no ground truth. Extras as in `bgloc.py`; `FEATSIM_TAG=_x` saves `featsim_x.npz` |
   | `bgloc.py C [extras]` | 1, 5: token-pair slot sharing by distance (background, same object, same class, different class). Extras are `name=pattern` with `{s}` for the seed suffix (e.g. `fin=trm_final{s}`); `BGLOC_TAG=_x` saves `bgloc_x.npz` |
   | `curve_an.py`, `curve_sa_an.py`, `curve_cmp.py` | 4, 10: learning curves, slot-attention collapse, the 1-step control |
   | `locprior.py` | 5, 6: locality imposed on T=1 by group |
   | `iter_profile.py C [sets]` | 3, 10: token-pair profile after each step (T=1 / T=8 iterated at test time by default; `sets` = `"name=rx file pattern:steps;..."`) |
   | `ctrl_an.py C runs` | 10: controls against T=1 and T=8 |
   | `read_drivers.py C runs` (GPU) | 3: does each slot's cross-attention follow appearance or position (regression per slot) |
   | `mech_reads.py C` | 3: read overlap between slots; does overlap predict convergence when T=1 is iterated |
   | `fresh_decoder.py C run [epochs]` (GPU) | 3: train a fresh decoder on a model's frozen final slots → `rx_{run}_fd.npz` |
   | `sel_curve.py C "file:run,...;..."` | 4: class selectivity and token-pair sharing at every saved epoch, from `curve*.npz` masks → `sel_curve{SEL_TAG}.npz` |
   | `sa_curve_any.py C file,...` | 4: slot attention's own map (FG-ARI, largest slot, slots used) at every epoch, from any `curve_sa*.npz` → `sa_curve_any.npz` |
   | `handover.py C runs epochs [tag]` (GPU) | 4: the output's dependence on slot attention's slots (swap them between images), slot attention's slot concentration, and its training gradient (size, coherence across batches) at each epoch → `handover{tag}.npz`; `voc:<run>` reads VOC-trained checkpoints |
   | `handover_an.py C files`, `handover_sum.py C` | 4: tables from `handover*.npz` |
   | `fig_mech.py C OUT` | `recursion_groups.png`, `recursion_dose.png` |
   | `fig_profile.py C OUT` | `recursion_profile.png` |
   | `fig_examples.py C OUT` | `recursion_examples.png` |
   | `fig_readme.py C results/figures` | `readme_examples.png`: the main README's two examples, chosen for clarity |
   | `fig_selectivity.py C OUT` | `recursion_selectivity.png`: reconstruction error against class selectivity and long-range locality (5) |
   | `fig_race.py C OUT` | `recursion_race.png`: the output's dependence on slot attention's slots, slot attention's own map and class selectivity over training (4) |

`post_control.sh <run>` waits for a control run (`scripts/runs/run_coco_recursion_controls.sh`) to finish, then runs `gpu_batch.py` and `mech_ctrl.py` on it.

The untied and wide controls (`trm_untied*`, `trm_wide*`) use the same scripts:
- `gpu_batch.py` and `mech_ctrl.py` on each seed;
- `ctrl_an.py C trm_untied` (or `trm_wide`);
- `bgloc.py` and `featsim.py` with the extra `unt=trm_untied{s}` (or `wide=trm_wide{s}`);
- at epochs 0–5, 7, 11 and 29: `handover.py`, `curve.py`, `curve_sa.py`, `sel_curve.py` and `sa_curve_any.py`.

`ws_process.sh <run>` scores a finished warm start (`scripts/runs/run_coco_warmstart.sh`): `gpu_batch.py` and `mech_ctrl.py` on all images, then per-epoch curves of the output and of slot attention's map (`curve.py`, `curve_sa.py`, `sel_curve.py`, `sa_curve_any.py`) and the per-step error. `peek.sh <run> <epochs>` does the per-epoch part for a run still training.

Earlier analyses that the report supersedes, kept for reference: `pairs.py`, `pairs3.py`, `mates.py`, `absorb.py`, `dumb.py`, `report.py`, `report2.py`, `sa_an.py`, `sa_report.py`, `ca_an.py`, `xm.py`, `dyn.py`, `recon.py`.
