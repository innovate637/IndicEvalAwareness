"""Measured discordance (G8) -> power -> the 200-vs-500 decision.

Replaces plan.md's assumed-discordance table. Two corrections over rev 1:
  * the full paired-difference variance is used, not just the null form;
  * English and Indic contrasts share all 200 items, so they are CORRELATED.
    rho=0 (independence) is CONSERVATIVE for power; report both.
"""
from __future__ import annotations
import json, math, random
from pathlib import Path
from phase2.config import load_run

run = load_run()
Z = 1.959964


def _draw(rng: random.Random, n: int, p10: float, p01: float,
          shared: list[float] | None = None) -> tuple[int, int]:
    b = c = 0
    for i in range(n):
        u = rng.random() if shared is None else shared[i]
        if u < p10:
            b += 1
        elif u < p10 + p01:
            c += 1
    return b, c


def mcnemar_power(n_items: int, p10: float, p01: float, n_sim: int = 20000,
                  alpha: float = 0.05, seed: int = 2026) -> float:
    """Exact/binomial McNemar power for one within-item cue contrast.
    p = min(1, 2 * BinomCDF(min(b,c); b+c, 0.5))."""
    rng = random.Random(seed)
    hits = 0
    for _ in range(n_sim):
        b, c = _draw(rng, n_items, p10, p01)
        n = b + c
        if n == 0:
            continue
        k = min(b, c)
        p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
        hits += int(p < alpha)
    return hits / n_sim


def paired_se(b: int, c: int, n: int) -> float:
    """FULL paired-difference SE: sqrt(b + c - (b-c)^2/n) / n.
    The (b+c)/n^2 form is only the null approximation (N12)."""
    v = max(0.0, (b + c) - (b - c) ** 2 / n)
    return math.sqrt(v) / n


def interaction_power(n_items: int, p10_en: float, p01_en: float,
                      p10_ix: float, p01_ix: float, rho: float = 0.0,
                      n_sim: int = 20000, alpha: float = 0.05,
                      seed: int = 2026) -> float:
    """Power for the difference-of-differences (English vs an Indic language).

    rho in [0,1] is the share of item-level randomness common to both
    languages. rho=0 => independent (conservative). Because the two contrasts
    use the SAME items, the true rho is > 0, positive covariance shrinks
    Var(d1-d2), and real power is HIGHER than the rho=0 figure.
    """
    rng = random.Random(seed)
    hits = 0
    for _ in range(n_sim):
        common = [rng.random() for _ in range(n_items)]
        if rho >= 1.0:
            u_en = u_ix = common
        else:
            u_en = [rho * common[i] + (1 - rho) * rng.random()
                    for i in range(n_items)]
            u_ix = [rho * common[i] + (1 - rho) * rng.random()
                    for i in range(n_items)]
        b1, c1 = _draw(rng, n_items, p10_en, p01_en, u_en)
        b2, c2 = _draw(rng, n_items, p10_ix, p01_ix, u_ix)
        d1, d2 = (c1 - b1) / n_items, (c2 - b2) / n_items
        se = math.sqrt(paired_se(b1, c1, n_items) ** 2
                       + paired_se(b2, c2, n_items) ** 2)
        if se <= 0:
            continue
        hits += int(abs(d1 - d2) / se > Z)
    return hits / n_sim


def decide(g8_path: Path, target_effect_pp: float = 10.0) -> dict:
    cells = json.loads(Path(g8_path).read_text())["per_cell"]
    disc = [v["discordance"] for v in cells.values() if v["n_pairs"] >= 20]
    if not disc:
        raise SystemExit("G8 has no cell with >=20 usable pairs -- the gate "
                         "refusal lexicons are missing (see gates.py).")
    d = sum(disc) / len(disc)
    eff = target_effect_pp / 100.0
    p01 = min(d, (d + eff) / 2)
    p10 = max(0.0, p01 - eff)
    rep = {"mean_discordance": round(d, 3), "assumed_p01": round(p01, 3),
           "assumed_p10": round(p10, 3), "target_effect_pp": target_effect_pp,
           "realised_effect_pp": round((p01 - p10) * 100, 1)}
    for n in (200, 300, 500):
        rep[f"main_power_n{n}"] = round(mcnemar_power(n, p10, p01), 3)
        rep[f"interaction_power_n{n}_rho0.0"] = round(
            interaction_power(n, p10, p01, d / 2, d / 2, rho=0.0), 3)
        rep[f"interaction_power_n{n}_rho0.5"] = round(
            interaction_power(n, p10, p01, d / 2, d / 2, rho=0.5), 3)
    ip200 = rep["interaction_power_n200_rho0.0"]
    ip500 = rep["interaction_power_n500_rho0.0"]
    rep["recommendation"] = (
        "STAY AT 200 -- H3 confirmatory" if ip200 >= 0.80 else
        "EXTEND TO 500 if translation capacity allows; else 200 with H3 "
        "pre-registered as EXPLORATORY" if ip500 >= 0.80 else
        "STAY AT 200 -- H3 is EXPLORATORY at any feasible n; report the "
        "interaction estimate with its CI and an explicit power statement")
    rep["note"] = ("rho=0 is conservative because the contrasts share items. "
                   "For REPORTED CIs use an item-level bootstrap, not "
                   "sqrt(v1+v2) (N11).")
    return rep


if __name__ == "__main__":
    import sys
    p = (Path(sys.argv[1]) if len(sys.argv) > 1
         else Path(run.paths["preflight"]) / "g8.json")
    tgt = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
    rep = decide(p, tgt)
    (Path(run.paths["preflight"]) / "power.json").write_text(
        json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
