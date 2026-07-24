#!/usr/bin/env python3
"""
31 — Cosine( -eval_awareness_dir , jailbreak_dir ): is the "unmonitored" state the same internal
configuration as the "jailbroken" state?  (collaborator A. Singh's hypothesis.)

CONFOUND CONTROL — this is the load-bearing detail:
  * eval_dir is a DIFFERENCE direction (diff-of-means: eval_log - neutral_log, the anti-trivial
    eval-framing contrast). Comparing it to a RAW mean jailbreak activation is invalid: a raw
    activation is dominated by the common-mode residual (huge shared norm), so the cosine would
    just measure the angle to the global mean activation, not anything jailbreak-specific.
  * So we compare DIFFERENCE-to-DIFFERENCE:  d_jail = mean(comply) - mean(refuse), and we build it
    WITHIN each jailbreak template (then average across templates) so template composition cannot
    leak into the direction. Both sides are then mean-centred by construction.

  cos( -eval_dir[L] , d_jail[L] )  > 0  ==> "not evaluated/unmonitored" aligns with "jailbroken/comply".

We report the FULL layer profile (0..39) with bootstrap 95% CIs, a random-direction floor, the value
at the eval-framing best layer (held-out AUROC peak), and the cross-lingual versions (en/hi/bn).

  python scripts/31_cosine_jail.py
Outputs: results/jailbreak/cosine_by_layer.csv , results/jailbreak/cosine_summary.csv , report/figs/fig_cosine_jail.pdf
"""
import os, sys, pickle, argparse
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import config

ACT_DIR = config.RESULTS_DIR / "activations"
JB_DIR  = config.JAILBREAK_DIR if hasattr(config, "JAILBREAK_DIR") else (config.RESULTS_DIR / "jailbreak")
NL = config.NUM_LAYERS
LANGS = ["en", "hi", "bn"]
MIN_PER_CELL = 3        # min comply (and min refuse) within a template to use that template's diff
N_BOOT = 2000
RNG = np.random.default_rng(0)


def unit(v, axis=-1, eps=1e-8):
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, eps)


def eval_dir_all_layers(lang):
    """eval-framing direction per layer: mean(eval_log) - mean(neutral_log).  -> [NL, 5120]"""
    e = pickle.load(open(ACT_DIR / f"{lang}_eval_log.pkl", "rb"))["acts"]      # [N,NL,5120]
    n = pickle.load(open(ACT_DIR / f"{lang}_neutral_log.pkl", "rb"))["acts"]
    return e.mean(0) - n.mean(0)                                                # [NL,5120]


def jail_dir_within_template(acts, refusal, templates, boot_idx=None):
    """d_jail[L] = template-averaged (mean(comply) - mean(refuse)), per layer.  -> [NL,5120] or None.
    boot_idx: optional dict {template: (comply_idx, refuse_idx)} of resampled positions for bootstrap."""
    acts = np.asarray(acts)                                                     # [N,NL,5120]
    refusal = np.asarray(refusal)
    templates = np.asarray(templates)
    diffs = []
    for t in np.unique(templates):
        m = templates == t
        comply = np.where(m & (refusal == 0))[0]
        refuse = np.where(m & (refusal == 1))[0]
        if len(comply) < MIN_PER_CELL or len(refuse) < MIN_PER_CELL:
            continue
        if boot_idx is not None:
            comply = RNG.choice(comply, len(comply), replace=True)
            refuse = RNG.choice(refuse, len(refuse), replace=True)
        diffs.append(acts[comply].mean(0) - acts[refuse].mean(0))              # [NL,5120]
    if not diffs:
        return None
    return np.mean(diffs, axis=0)                                              # [NL,5120]


def cosine_profile(eval_d, jail_d):
    """per-layer cos(-eval_dir, d_jail)."""
    return np.sum(unit(-eval_d) * unit(jail_d), axis=1)                        # [NL]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=LANGS)
    a = ap.parse_args()

    # eval-framing best layer (held-out AUROC peak) per lang, for the headline number
    au = pd.read_csv(config.PROBE_DIR / "auroc_all.csv")
    au = au[au.contrast == "eval_framing"]
    best_layer = {l: int(au[au.lang == l].sort_values("auroc", ascending=False).iloc[0]["layer"])
                  for l in au.lang.unique()}

    rows_prof, rows_sum = [], []
    for lang in a.langs:
        jp = JB_DIR / f"{lang}_jailbreak.pkl"
        if not jp.exists():
            print(f"[{lang}] no jailbreak pkl — skip"); continue
        jb = pickle.load(open(jp, "rb"))
        refusal = np.asarray(jb["refusal"])
        keep = refusal != None                                                 # drop unjudged
        acts = np.asarray(jb["acts"])[keep]
        refusal = refusal[keep].astype(int)
        templates = np.asarray(jb["template"])[keep]
        nC, nR = int((refusal == 0).sum()), int((refusal == 1).sum())

        eval_d = eval_dir_all_layers(lang)                                     # [NL,5120]
        jail_d = jail_dir_within_template(acts, refusal, templates)
        if jail_d is None:
            print(f"[{lang}] no template has >={MIN_PER_CELL} comply & >={MIN_PER_CELL} refuse "
                  f"(comply={nC}) — skip"); continue
        cos = cosine_profile(eval_d, jail_d)                                   # [NL]

        # bootstrap CI per layer (resample comply & refuse within each template)
        boots = np.empty((N_BOOT, NL), np.float32)
        for b in range(N_BOOT):
            jd = jail_dir_within_template(acts, refusal, templates, boot_idx=True)
            boots[b] = cosine_profile(eval_d, jd)
        lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)

        # random-direction floor: cos(random unit, d_jail) magnitude
        rfloor = []
        for _ in range(200):
            r = unit(RNG.standard_normal((NL, eval_d.shape[1])))
            rfloor.append(np.abs(np.sum(r * unit(jail_d), axis=1)))
        rfloor = np.percentile(np.concatenate(rfloor), 95)                     # 95th pct |cos| under random

        for L in range(NL):
            rows_prof.append(dict(lang=lang, layer=L, cos=cos[L], lo=lo[L], hi=hi[L],
                                  sig=int(lo[L] > 0 or hi[L] < 0)))
        bl = best_layer.get(lang, int(np.argmax(np.abs(cos))))
        rows_sum.append(dict(lang=lang, n_comply=nC, n_refuse=nR,
                             best_layer=bl, cos_at_best=cos[bl], lo_best=lo[bl], hi_best=hi[bl],
                             peak_layer=int(np.argmax(cos)), cos_peak=cos.max(),
                             random_floor95=rfloor))
        print(f"[{lang}] comply={nC} refuse={nR} | best L{bl}: cos={cos[bl]:+.3f} "
              f"[{lo[bl]:+.3f},{hi[bl]:+.3f}] | peak L{int(np.argmax(cos))}={cos.max():+.3f} "
              f"| random floor(95%)={rfloor:.3f}", flush=True)

    prof = pd.DataFrame(rows_prof); summ = pd.DataFrame(rows_sum)
    JB_DIR.mkdir(parents=True, exist_ok=True)
    prof.to_csv(JB_DIR / "cosine_by_layer.csv", index=False)
    summ.to_csv(JB_DIR / "cosine_summary.csv", index=False)
    print("\nSUMMARY\n" + summ.round(3).to_string(index=False))

    # figure: layer profile with CI band per lang
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        FIG = ROOT / "report" / "figs"; FIG.mkdir(parents=True, exist_ok=True)
        col = {"en": "#13315c", "hi": "#d97706", "bn": "#7c3aed"}
        fig, ax = plt.subplots(figsize=(7, 3.8))
        for lang in a.langs:
            p = prof[prof.lang == lang]
            if p.empty: continue
            ax.plot(p.layer, p.cos, color=col.get(lang, "#333"), lw=1.8, label=lang.upper())
            ax.fill_between(p.layer, p.lo, p.hi, color=col.get(lang, "#333"), alpha=0.15)
            bl = int(summ[summ.lang == lang].iloc[0]["best_layer"])
            ax.scatter([bl], [p[p.layer == bl].cos.iloc[0]], color=col.get(lang, "#333"), s=30, zorder=5)
        ax.axhline(0, color="#999", lw=0.8)
        ax.set_xlabel("layer"); ax.set_ylabel(r"cos($-$eval_dir, jailbreak_dir)")
        ax.set_title("Is the unmonitored state the jailbroken state? (dot = eval-framing best layer)")
        ax.legend(frameon=False, fontsize=9)
        fig.savefig(FIG / "fig_cosine_jail.pdf", bbox_inches="tight"); plt.close()
        print(f"figure -> {FIG/'fig_cosine_jail.pdf'}")
    except Exception as e:
        print("fig skipped:", e)


if __name__ == "__main__":
    main()
