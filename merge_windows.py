#!/usr/bin/env python3
"""
merge_windows.py — assemble the window-level master table.

Joins, on the shared non-overlapping window grid:

  * SynDiv per population        (SynDiv_c window output, one file per subset)
  * Syn-Fst per population pair  (window-averaged output of 04_syn_fst.sbatch)
  * individual-level synteny     (synteny_freq_*.tsv.gz from 05)

and derives:

  * DELTA_<POP>     population SynDiv minus the 77-accession SynDiv
  * PAVFST_<A>_<B>  Hudson Fst on synteny presence/absence, an allele-frequency
                    style companion to SynDiv's Syn-Fst
  * genome-wide and per-chromosome summaries, plus the 6x6 mean Fst matrix

Everything is keyed on (CHROM, START, END); files that are missing are skipped
with a warning so partial runs still produce a usable table.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd

KEY = ["CHROM", "START", "END"]


def warn(msg):
    print(f"[merge_windows] WARNING: {msg}", file=sys.stderr)


def info(msg):
    print(f"[merge_windows] {msg}", file=sys.stderr)


def read_win(path, value_name):
    """Read a SynDiv_c window output into CHROM/START/END/<value_name>."""
    df = pd.read_csv(path, sep="\t")
    if df.shape[1] != 4:
        raise SystemExit(f"unexpected window file layout ({df.shape[1]} columns): {path}")
    df.columns = KEY + [value_name]
    df["CHROM"] = df["CHROM"].astype(str)
    df["START"] = df["START"].astype(np.int64)
    df["END"] = df["END"].astype(np.int64)
    return df


def read_fst_win(path, value_name):
    df = pd.read_csv(path, sep="\t")
    df = df.rename(columns={df.columns[0]: "CHROM"})
    df["CHROM"] = df["CHROM"].astype(str)
    keep = df[KEY + ["MEAN_FST", "MAX_FST", "N_BASES"]].copy()
    keep = keep.rename(columns={"MEAN_FST": value_name,
                                "MAX_FST": value_name + "_MAX",
                                "N_BASES": value_name + "_N"})
    return keep


def hudson_fst(p1, p2, n1, n2):
    """Hudson's Fst estimator for a biallelic frequency (here: synteny present)."""
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    num = (p1 - p2) ** 2
    if n1 > 1:
        num = num - p1 * (1 - p1) / (n1 - 1)
    if n2 > 1:
        num = num - p2 * (1 - p2) / (n2 - 1)
    den = p1 * (1 - p2) + p2 * (1 - p1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(den > 0, num / den, np.nan)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cal-dir", required=True, help="directory with <TAG>.<grid>.win.out")
    ap.add_argument("--fst-dir", required=True, help="directory with win/<A>_vs_<B>.<grid>.tsv.gz")
    ap.add_argument("--freq", default="", help="synteny_freq_<grid>.tsv.gz from 05 (individual level)")
    ap.add_argument("--pops", required=True, help="comma-separated population order")
    ap.add_argument("--pop-sizes", required=True, help="pop_colors.tsv (pop, label, color, n)")
    ap.add_argument("--grid", default="w5k", help="grid tag used in the file names [w5k]")
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()

    pops = [p for p in args.pops.split(",") if p]
    sizes = pd.read_csv(args.pop_sizes, sep="\t")
    n_of = dict(zip(sizes["pop"], sizes["n"]))

    # ------------------------------------------------ population SynDiv
    all_path = os.path.join(args.cal_dir, f"ALL77.{args.grid}.win.out")
    if not os.path.exists(all_path):
        sys.exit(f"missing {all_path} — run 03b_all77_windows.sbatch")
    master = read_win(all_path, "SYNDIV_ALL77")
    info(f"ALL77: {len(master):,} windows")

    for p in pops:
        path = os.path.join(args.cal_dir, f"{p}.{args.grid}.win.out")
        if not os.path.exists(path):
            warn(f"no SynDiv window file for {p} ({path})")
            continue
        master = master.merge(read_win(path, f"SYNDIV_{p}"), on=KEY, how="left")

    for p in pops:
        col = f"SYNDIV_{p}"
        if col in master.columns:
            master[f"DELTA_{p}"] = master[col] - master["SYNDIV_ALL77"]

    master["WIN_BP"] = master["END"] - master["START"] + 1

    # ------------------------------------------------ individual-level synteny
    if args.freq and os.path.exists(args.freq):
        freq = pd.read_csv(args.freq, sep="\t")
        freq = freq.rename(columns={freq.columns[0]: "CHROM"})
        freq["CHROM"] = freq["CHROM"].astype(str)
        drop = [c for c in ("WIN_BP", "N_ACC") if c in freq.columns]
        master = master.merge(freq.drop(columns=drop), on=KEY, how="left")
        info(f"joined individual-level synteny frequencies from {os.path.basename(args.freq)}")
    elif args.freq:
        warn(f"missing {args.freq} — individual-level columns skipped")

    # ------------------------------------------------ Syn-Fst per pair
    pairs = list(itertools.combinations(pops, 2))
    contrasts = [(a, b, f"{a}_vs_{b}") for a, b in pairs] + [(p, f"not{p}", f"{p}_vs_not{p}") for p in pops]
    n_fst = 0
    for a, b, tag in contrasts:
        path = os.path.join(args.fst_dir, "win", f"{tag}.{args.grid}.tsv.gz")
        if not os.path.exists(path):
            warn(f"no windowed Fst for {tag}")
            continue
        master = master.merge(read_fst_win(path, f"FST_{tag}"), on=KEY, how="left")
        n_fst += 1
    info(f"joined {n_fst}/{len(contrasts)} Syn-Fst contrasts")

    # ------------------------------------------------ presence/absence Fst
    for a, b in pairs:
        ca, cb = f"FREQ_{a}", f"FREQ_{b}"
        if ca in master.columns and cb in master.columns:
            master[f"PAVFST_{a}_{b}"] = hudson_fst(master[ca], master[cb], n_of.get(a, 0), n_of.get(b, 0))

    master = master.sort_values(["CHROM", "START"]).reset_index(drop=True)
    out_master = f"{args.out_prefix}.tsv.gz"
    master.to_csv(out_master, sep="\t", index=False, float_format="%.6g", na_rep="NA", compression="gzip")
    info(f"wrote {out_master}  ({master.shape[0]:,} windows x {master.shape[1]} columns)")

    # ------------------------------------------------ genome-wide summaries
    rows = []
    for p in ["ALL77"] + pops:
        col = f"SYNDIV_{p}"
        if col not in master.columns:
            continue
        v = master[col].to_numpy(dtype=float)
        bp = master["WIN_BP"].to_numpy(dtype=float)
        ok = ~np.isnan(v)
        rows.append({
            "pop": p,
            "n_accessions": int(n_of.get(p, 77 if p == "ALL77" else 0)),
            "n_windows": int(ok.sum()),
            "mean_syndiv": float(np.average(v[ok], weights=bp[ok])) if ok.any() else np.nan,
            "median_syndiv": float(np.median(v[ok])) if ok.any() else np.nan,
            "sd_syndiv": float(np.std(v[ok], ddof=1)) if ok.sum() > 1 else np.nan,
            "q05": float(np.quantile(v[ok], 0.05)) if ok.any() else np.nan,
            "q95": float(np.quantile(v[ok], 0.95)) if ok.any() else np.nan,
        })
    pd.DataFrame(rows).to_csv(f"{args.out_prefix}.pop_summary.tsv", sep="\t", index=False, float_format="%.6f")
    info(f"wrote {args.out_prefix}.pop_summary.tsv")

    # per chromosome
    chr_rows = []
    for chrom, sub in master.groupby("CHROM"):
        for p in ["ALL77"] + pops:
            col = f"SYNDIV_{p}"
            if col not in sub.columns:
                continue
            v = sub[col].to_numpy(dtype=float)
            ok = ~np.isnan(v)
            chr_rows.append({"chrom": chrom, "pop": p, "n_windows": int(ok.sum()),
                             "mean_syndiv": float(v[ok].mean()) if ok.any() else np.nan})
    pd.DataFrame(chr_rows).to_csv(f"{args.out_prefix}.pop_chrom_summary.tsv", sep="\t", index=False, float_format="%.6f")

    # ------------------------------------------------ 6x6 mean Fst matrices
    for prefix, label in (("FST", "synfst"), ("PAVFST", "pavfst")):
        mat = pd.DataFrame(np.nan, index=pops, columns=pops, dtype=float)
        for a, b in pairs:
            col = f"{prefix}_{a}_vs_{b}" if prefix == "FST" else f"{prefix}_{a}_{b}"
            if col in master.columns:
                m = float(np.nanmean(master[col].to_numpy(dtype=float)))
                mat.loc[a, b] = m
                mat.loc[b, a] = m
        np.fill_diagonal(mat.values, 0.0)
        out = f"{args.out_prefix}.{label}_matrix.tsv"
        mat.to_csv(out, sep="\t", float_format="%.6f")
        info(f"wrote {out}")


if __name__ == "__main__":
    main()
