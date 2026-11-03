"""Table of handover.py results: reliance on slot attention's slots, slot attention's slot
concentration, gradient size and coherence, and the push towards concentration.
Usage: handover_an.py C file1,file2 [runs]"""
import sys

import numpy as np

C = sys.argv[1]
R = {}
for f in sys.argv[2].split(","):
    Z = np.load(f"{C}/{f}")
    R.update({k: Z[k] for k in Z.files})
runs = sys.argv[3].split(",") if len(sys.argv) > 3 else sorted({k.split("|")[0] for k in R})


def t(a):
    return a.mean() / (a.std(ddof=1) / np.sqrt(len(a)))


print("run|epoch  rel  conc  |dL/dx|  rms sa (coh)   rms proj via sa (coh)  via module (coh)  push sa t / proj via sa t / via module t")
for r in runs:
    eps = sorted({int(k.split("|")[1][1:]) for k in R if k.startswith(r + "|")})
    for e in eps:
        k = f"{r}|e{e}"
        m, mr, mo = R[f"{k}|err"][:, -1].mean(), R[f"{k}|err_roll"][:, -1].mean(), R[f"{k}|err_other"].mean()
        rel = (mr - m) / (mo - m)
        print(f"{r}|e{e:<3d} {rel:5.2f} {float(R[f'{k}|conc']):.3f} {R[f'{k}|gx'].mean():.1e}  "
              f"{np.sqrt(R[f'{k}|g_sa_sq']):.2e} ({float(R[f'{k}|g_sa_coh']):+.3f})  "
              f"{np.sqrt(R[f'{k}|g_proj_viasa_sq']):.2e} ({float(R[f'{k}|g_proj_viasa_coh']):+.3f})  "
              f"{np.sqrt(R[f'{k}|g_proj_viamod_sq']):.2e} ({float(R[f'{k}|g_proj_viamod_coh']):+.3f})  "
              f"{t(R[f'{k}|push_sa']):+5.1f} / {t(R[f'{k}|push_proj_viasa']):+5.1f} / {t(R[f'{k}|push_proj_viamod']):+5.1f}")
