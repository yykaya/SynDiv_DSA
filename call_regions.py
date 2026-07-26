#!/usr/bin/env python3
"""
call_regions.py — locus level: high/low synteny regions, Fst peaks, and the
overlap statistics between them.

Terminology (SynDiv is a *diversity*, so the sign is easy to get backwards):

    SYNDIV high  ->  little shared synteny   ->  LSR, "low-synteny region"
    SYNDIV low   ->  synteny conserved       ->  HSR, "high-synteny region"

Region calls (all on the non-overlapping window grid of the master table):

    HSR_ALL77 / LSR_ALL77          panel-wide tails of SYNDIV_ALL77
    HSR_<POP> / LSR_<POP>          the same per population
    DIVERGENT_<POP>                DELTA_<POP> in the upper tail: this
                                   population is far less syntenic here than
                                   the panel as a whole
    CONSERVED_<POP>                DELTA_<POP> in the lower tail
    FSTPEAK_<A>_vs_<B>             upper tail of the windowed Syn-Fst

Adjacent qualifying windows are merged (a gap of up to --max-gap windows is
bridged) and regions shorter than --min-len bp are dropped.

Also writes region_summary.tsv and synteny_vs_fst_enrichment.tsv, the
"are high-Fst loci also low-synteny loci?" contingency analysis.
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import sys

import numpy as np
import pandas as pd


def info(msg):
    print(f"[call_regions] {msg}", file=sys.stderr)


def merge_windows(df, mask, max_gap_windows, min_len):
    """Merge flagged rows of a per-chromosome-sorted window table into regions."""
    out = []
    for chrom, sub in df.loc[mask, ["CHROM", "START", "END", "_SCORE"]].groupby("CHROM", sort=False):
        sub = sub.sort_values("START")
        starts = sub["START"].to_numpy()
        ends = sub["END"].to_numpy()
        scores = sub["_SCORE"].to_numpy(dtype=float)
        if starts.size == 0:
            continue
        win = int(np.median(ends - starts + 1)) or 1
        cs, ce = starts[0], ends[0]
        acc = [scores[0]]
        for s, e, sc in zip(starts[1:], ends[1:], scores[1:]):
            if s - ce - 1 <= max_gap_windows * win:
                ce = max(ce, e)
                acc.append(sc)
            else:
                out.append((chrom, cs, ce, len(acc), float(np.mean(acc)), float(np.max(acc))))
                cs, ce, acc = s, e, [sc]
        out.append((chrom, cs, ce, len(acc), float(np.mean(acc)), float(np.max(acc))))
    regions = pd.DataFrame(out, columns=["CHROM", "START", "END", "N_WINDOWS", "MEAN_SCORE", "MAX_SCORE"])
    if len(regions):
        regions["LENGTH"] = regions["END"] - regions["START"] + 1
        regions = regions[regions["LENGTH"] >= min_len].reset_index(drop=True)
    else:
        regions["LENGTH"] = []
    return regions


def write_bed(regions, path, name):
    """BED6-ish: chrom, start0, end, name, score, strand."""
    with open(path, "w") as fh:
        for i, r in regions.iterrows():
            fh.write(f"{r.CHROM}\t{int(r.START) - 1}\t{int(r.END)}\t{name}_{i + 1}\t{r.MEAN_SCORE:.6f}\t.\n")


def log_binom(n, k):
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def fisher_greater(a, b, c, d):
    """One-sided (enrichment) Fisher exact p for the 2x2 table [[a,b],[c,d]]."""
    n = a + b + c + d
    row1, col1 = a + b, a + c
    lo = max(0, col1 - (n - row1))
    hi = min(row1, col1)
    denom = log_binom(n, col1)
    p = 0.0
    for x in range(int(a), int(hi) + 1):
        lp = log_binom(row1, x) + log_binom(n - row1, col1 - x) - denom
        p += math.exp(lp)
    return min(1.0, max(0.0, p)), lo, hi


def odds_ratio(a, b, c, d):
    a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5   # Haldane-Anscombe
    return (a * d) / (b * c)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--master", required=True, help="master_w5k.tsv.gz from 06")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--pops", required=True)
    ap.add_argument("--tail", type=float, default=0.05, help="tail fraction for HSR/LSR [0.05]")
    ap.add_argument("--fst-tail", type=float, default=0.01, help="tail fraction for Fst peaks [0.01]")
    ap.add_argument("--delta-tail", type=float, default=0.05, help="tail fraction for population-specific regions [0.05]")
    ap.add_argument("--max-gap", type=int, default=1, help="bridge this many missing windows when merging [1]")
    ap.add_argument("--min-len", type=int, default=10000, help="minimum region length in bp [10000]")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    pops = [p for p in args.pops.split(",") if p]

    df = pd.read_csv(args.master, sep="\t", na_values=["NA"])
    df["CHROM"] = df["CHROM"].astype(str)
    df = df.sort_values(["CHROM", "START"]).reset_index(drop=True)
    info(f"{len(df):,} windows, {df.shape[1]} columns")

    summary = []
    calls = {}          # label -> boolean mask, kept for the contingency step

    def call(label, column, side, tail, kind):
        if column not in df.columns:
            info(f"skip {label}: no column {column}")
            return
        v = df[column].to_numpy(dtype=float)
        ok = ~np.isnan(v)
        if ok.sum() == 0:
            info(f"skip {label}: column {column} is empty")
            return
        thr = np.quantile(v[ok], 1 - tail if side == "high" else tail)
        mask = ok & ((v >= thr) if side == "high" else (v <= thr))
        df["_SCORE"] = np.where(np.isnan(v), 0.0, v)
        regions = merge_windows(df, mask, args.max_gap, args.min_len)
        bed = os.path.join(args.outdir, f"{label}.bed")
        write_bed(regions, bed, label)
        calls[label] = mask
        total_bp = int(regions["LENGTH"].sum()) if len(regions) else 0
        summary.append({"region_set": label, "kind": kind, "column": column, "side": side,
                        "tail": tail, "threshold": thr, "n_windows": int(mask.sum()),
                        "n_regions": len(regions), "total_bp": total_bp,
                        "bed": os.path.basename(bed)})
        info(f"{label:<26} thr={thr:.4f}  windows={int(mask.sum()):>6}  regions={len(regions):>5}  bp={total_bp:,}")

    # ------------------------------------------------------ panel-wide tails
    call("LSR_ALL77", "SYNDIV_ALL77", "high", args.tail, "low_synteny")
    call("HSR_ALL77", "SYNDIV_ALL77", "low", args.tail, "high_synteny")

    # ------------------------------------------------------ per population
    for p in pops:
        call(f"LSR_{p}", f"SYNDIV_{p}", "high", args.tail, "low_synteny")
        call(f"HSR_{p}", f"SYNDIV_{p}", "low", args.tail, "high_synteny")
        call(f"DIVERGENT_{p}", f"DELTA_{p}", "high", args.delta_tail, "pop_divergent")
        call(f"CONSERVED_{p}", f"DELTA_{p}", "low", args.delta_tail, "pop_conserved")

    # ------------------------------------------------------ Fst peaks
    contrasts = [f"{a}_vs_{b}" for a, b in itertools.combinations(pops, 2)] + [f"{p}_vs_not{p}" for p in pops]
    for tag in contrasts:
        call(f"FSTPEAK_{tag}", f"FST_{tag}", "high", args.fst_tail, "fst_peak")

    pd.DataFrame(summary).to_csv(os.path.join(args.outdir, "region_summary.tsv"),
                                 sep="\t", index=False, float_format="%.6f")

    # ------------------------------- do high-Fst windows sit in low-synteny DNA?
    rows = []
    for tag in contrasts:
        fcol = f"FST_{tag}"
        if fcol not in df.columns:
            continue
        peak = calls.get(f"FSTPEAK_{tag}")
        if peak is None:
            continue
        for syn_label in ["LSR_ALL77", "HSR_ALL77"]:
            syn = calls.get(syn_label)
            if syn is None:
                continue
            a = int(np.sum(peak & syn))
            b = int(np.sum(peak & ~syn))
            c = int(np.sum(~peak & syn))
            d = int(np.sum(~peak & ~syn))
            p, _, _ = fisher_greater(a, b, c, d)
            rows.append({"contrast": tag, "synteny_class": syn_label,
                         "peak_and_class": a, "peak_only": b, "class_only": c, "neither": d,
                         "odds_ratio": odds_ratio(a, b, c, d), "fisher_p_greater": p})

        # correlation between Syn-Fst and syntenic diversity across all windows
        for other in ["SYNDIV_ALL77"] + [f"SYNDIV_{p}" for p in pops]:
            if other not in df.columns:
                continue
            x = df[fcol].to_numpy(dtype=float)
            y = df[other].to_numpy(dtype=float)
            ok = ~np.isnan(x) & ~np.isnan(y)
            if ok.sum() > 10:
                r = float(np.corrcoef(x[ok], y[ok])[0, 1])
                rows.append({"contrast": tag, "synteny_class": f"corr:{other}",
                             "peak_and_class": int(ok.sum()), "peak_only": np.nan,
                             "class_only": np.nan, "neither": np.nan,
                             "odds_ratio": np.nan, "fisher_p_greater": np.nan, "pearson_r": r})

    enr = pd.DataFrame(rows)
    enr.to_csv(os.path.join(args.outdir, "synteny_vs_fst_enrichment.tsv"),
               sep="\t", index=False, float_format="%.6g", na_rep="NA")
    info(f"wrote {os.path.join(args.outdir, 'synteny_vs_fst_enrichment.tsv')}")

    # ------------------------------- one BED with every window and its classes
    cls = pd.DataFrame({"CHROM": df["CHROM"], "START": df["START"] - 1, "END": df["END"]})
    for label, mask in calls.items():
        cls[label] = mask.astype(int)
    cls.to_csv(os.path.join(args.outdir, "window_classes.tsv.gz"),
               sep="\t", index=False, compression="gzip")
    info("wrote window_classes.tsv.gz")


if __name__ == "__main__":
    main()
