#!/usr/bin/env python3
"""
window_stats.py — join per-window annotations onto the synteny master table and
test the associations.

Outputs
  master_w5k.annotated.tsv.gz   every synteny column plus every annotation column
  correlations.tsv              Pearson + Spearman of each annotation against
                                each synteny / Fst response
  class_contrasts.tsv           annotation levels inside vs outside each region
                                class (LSR, HSR, DIVERGENT_<POP>, FST peaks ...)
                                with a Mann-Whitney test

No SciPy needed: ranks, the Spearman coefficient and the Mann-Whitney normal
approximation (tie-corrected) are computed here.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import pandas as pd


def info(msg):
    print(f"[window_stats] {msg}", file=sys.stderr)


def rankdata(x):
    """Average ranks, ties shared (same convention as scipy.stats.rankdata)."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=float)
    sx = x[order]
    i = 0
    while i < x.size:
        j = i
        while j + 1 < x.size and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def pearson(x, y):
    if x.size < 3:
        return np.nan
    sx, sy = x.std(), y.std()
    if sx == 0 or sy == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    if x.size < 3:
        return np.nan
    return pearson(rankdata(x), rankdata(y))


def norm_sf(z):
    """Upper tail of the standard normal."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def mannwhitney(a, b):
    """Two-sided Mann-Whitney U with the tie-corrected normal approximation."""
    n1, n2 = a.size, b.size
    if n1 < 3 or n2 < 3:
        return np.nan, np.nan
    allv = np.concatenate([a, b])
    r = rankdata(allv)
    R1 = r[:n1].sum()
    U1 = R1 - n1 * (n1 + 1) / 2.0
    mu = n1 * n2 / 2.0
    n = n1 + n2
    _, counts = np.unique(allv, return_counts=True)
    tie = float(np.sum(counts ** 3 - counts))
    var = n1 * n2 / 12.0 * ((n + 1) - tie / (n * (n - 1)))
    if var <= 0:
        return float(U1), np.nan
    z = (U1 - mu) / math.sqrt(var)
    return float(z), float(2.0 * norm_sf(abs(z)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--master", required=True)
    ap.add_argument("--annot", nargs="*", default=[], help="two-column-plus TSVs keyed on WINDOW")
    ap.add_argument("--classes", default="", help="window_classes.tsv.gz from 07")
    ap.add_argument("--pops", default="")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--min-class-windows", type=int, default=20)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    pops = [p for p in args.pops.split(",") if p]

    master = pd.read_csv(args.master, sep="\t", na_values=["NA"])
    master["CHROM"] = master["CHROM"].astype(str)
    master["WINDOW"] = master["CHROM"] + "_" + master["START"].astype(str)
    info(f"master: {master.shape[0]:,} windows x {master.shape[1]} columns")

    annot_cols = []
    for path in args.annot:
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            info(f"skip missing annotation block {path}")
            continue
        a = pd.read_csv(path, sep="\t")
        if "WINDOW" not in a.columns:
            info(f"skip {path}: no WINDOW column")
            continue
        a = a.drop_duplicates(subset="WINDOW")
        new = [c for c in a.columns if c != "WINDOW"]
        master = master.merge(a, on="WINDOW", how="left")
        annot_cols += new
    annot_cols = [c for c in dict.fromkeys(annot_cols) if c in master.columns]
    info(f"annotation columns: {len(annot_cols)}")

    classes = pd.DataFrame()
    if args.classes and os.path.exists(args.classes):
        classes = pd.read_csv(args.classes, sep="\t")
        classes["CHROM"] = classes["CHROM"].astype(str)
        classes["WINDOW"] = classes["CHROM"] + "_" + (classes["START"] + 1).astype(str)
        classes = classes.drop(columns=["CHROM", "START", "END"])
        master = master.merge(classes, on="WINDOW", how="left")
        info(f"region classes: {classes.shape[1] - 1}")

    out_master = os.path.join(args.outdir, os.path.basename(args.master).replace(".tsv.gz", ".annotated.tsv.gz"))
    master.to_csv(out_master, sep="\t", index=False, float_format="%.6g", na_rep="NA", compression="gzip")
    info(f"wrote {out_master}")

    # ------------------------------------------------------------ correlations
    responses = [c for c in master.columns
                 if c.startswith(("SYNDIV_", "DELTA_", "FREQ_", "MEAN_FRAC_", "PAVFST_"))
                 or (c.startswith("FST_") and not c.endswith(("_MAX", "_N")))]
    rows = []
    for resp in responses:
        y_all = master[resp].to_numpy(dtype=float)
        for ann in annot_cols:
            x_all = master[ann].to_numpy(dtype=float)
            ok = ~np.isnan(x_all) & ~np.isnan(y_all)
            if ok.sum() < 50:
                continue
            x, y = x_all[ok], y_all[ok]
            rows.append({"response": resp, "annotation": ann, "n_windows": int(ok.sum()),
                         "pearson_r": pearson(x, y), "spearman_rho": spearman(x, y),
                         "mean_annotation": float(x.mean())})
    corr = pd.DataFrame(rows)
    if len(corr):
        corr = corr.reindex(corr["spearman_rho"].abs().sort_values(ascending=False).index)
    corr.to_csv(os.path.join(args.outdir, "correlations.tsv"), sep="\t", index=False,
                float_format="%.6g", na_rep="NA")
    info(f"wrote correlations.tsv ({len(corr)} rows)")

    # ------------------------------------------------------- class contrasts
    class_cols = [c for c in master.columns
                  if c.startswith(("LSR_", "HSR_", "DIVERGENT_", "CONSERVED_", "FSTPEAK_"))]
    rows = []
    for cls in class_cols:
        m = master[cls].fillna(0).to_numpy() > 0
        if m.sum() < args.min_class_windows:
            continue
        for ann in annot_cols:
            v = master[ann].to_numpy(dtype=float)
            inside = v[m & ~np.isnan(v)]
            outside = v[~m & ~np.isnan(v)]
            if inside.size < args.min_class_windows or outside.size < args.min_class_windows:
                continue
            z, p = mannwhitney(inside, outside)
            mi, mo = float(inside.mean()), float(outside.mean())
            rows.append({"region_class": cls, "annotation": ann,
                         "n_inside": int(inside.size), "n_outside": int(outside.size),
                         "mean_inside": mi, "mean_outside": mo,
                         "log2_ratio": math.log2((mi + 1e-9) / (mo + 1e-9)),
                         "mw_z": z, "mw_p": p})
    cc = pd.DataFrame(rows)
    if len(cc):
        # Benjamini-Hochberg across all tests
        p = cc["mw_p"].to_numpy(dtype=float)
        ok = ~np.isnan(p)
        q = np.full(p.shape, np.nan)
        if ok.sum():
            idx = np.argsort(p[ok])
            ranked = p[ok][idx]
            n = ranked.size
            qv = ranked * n / (np.arange(n) + 1)
            qv = np.minimum.accumulate(qv[::-1])[::-1]
            tmp = np.empty(n)
            tmp[idx] = np.clip(qv, 0, 1)
            q[ok] = tmp
        cc["mw_q"] = q
        cc = cc.reindex(cc["mw_z"].abs().sort_values(ascending=False).index)
    cc.to_csv(os.path.join(args.outdir, "class_contrasts.tsv"), sep="\t", index=False,
              float_format="%.6g", na_rep="NA")
    info(f"wrote class_contrasts.tsv ({len(cc)} rows)")

    # ------------------------------------------------------------- highlights
    if len(corr):
        info("strongest annotation correlations with panel-wide syntenic diversity:")
        sub = corr[corr["response"] == "SYNDIV_ALL77"].head(12)
        print(sub.to_string(index=False), file=sys.stderr)


if __name__ == "__main__":
    main()
