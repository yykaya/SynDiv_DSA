#!/usr/bin/env python3
"""
aggregate_annot.py — turn the per-accession outputs of 08b plus the reference
annotation layer of 08a into window-level and accession-level tables.

Window level (all keyed on WINDOW = "<chrom>_<1-based window start>"):
  window_te_projection.tsv   TE fraction of the orthologous segment, mean over
                             all accessions and per population, plus how much
                             sequence the accessions actually carry there
  window_genes.tsv           gene count, gene bp, mean synteny conservation of
                             the genes in the window, counts per conservation class
  window_sv.tsv              SV count / bp / mean allele frequency, overall and
                             per population
  window_cen.tsv             centromere overlap fraction

Gene level:
  genes_ColPEK_synteny.tsv   per Col-PEK gene: in how many accessions (and in
                             which populations) the gene sits in a syntenic
                             block, plus a conservation class

Accession level:
  te_synteny_per_accession.tsv / te_synteny_per_pop.tsv
  individual_summary.tsv     synteny fraction, TE enrichment in the
                             non-syntenic part, gene counts, SV burden
"""

from __future__ import annotations

import argparse
import glob
import gzip
import os
import sys

import numpy as np
import pandas as pd


def info(msg):
    print(f"[aggregate_annot] {msg}", file=sys.stderr)


def warn(msg):
    print(f"[aggregate_annot] WARNING: {msg}", file=sys.stderr)


def read_pop_map(path):
    d = {}
    with open(path) as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) >= 2:
                d[f[0]] = f[1]
    return d


def window_id(chrom, pos, w):
    return f"{chrom}_{((pos - 1) // w) * w + 1}"


def spread_interval(acc, chrom, start0, end, w, value_per_bp=None, chrom_offsets=None):
    """Add an interval's bp (or a value) to every window it overlaps."""
    first = start0 // w
    last = (end - 1) // w
    for k in range(first, last + 1):
        ws, we = k * w, (k + 1) * w
        ov = min(end, we) - max(start0, ws)
        if ov > 0:
            acc[f"{chrom}_{ws + 1}"] = acc.get(f"{chrom}_{ws + 1}", 0) + (ov if value_per_bp is None else ov * value_per_bp)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--acc-dir", required=True, help="analysis/annot/per_accession")
    ap.add_argument("--ref-dir", required=True, help="analysis/annot/ref")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pop-map", required=True)
    ap.add_argument("--pops", required=True)
    ap.add_argument("--window", type=int, default=5000)
    ap.add_argument("--syn-per-accession", default="", help="per_accession_synteny.tsv from 05")
    ap.add_argument("--sv-per-accession", default="", help="sv_per_accession.tsv from 08a")
    ap.add_argument("--pancore", default="", help="optional gene_id <TAB> class table")
    ap.add_argument("--core-frac", type=float, default=0.95)
    ap.add_argument("--shell-frac", type=float, default=0.10)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    W = args.window
    pops = [p for p in args.pops.split(",") if p]
    pop_of = read_pop_map(args.pop_map)
    accs = sorted(pop_of)
    info(f"{len(accs)} accessions, window {W}")

    # ------------------------------------------------------------------ TE
    te_rows = {}
    n_te = 0
    for acc in accs:
        path = os.path.join(args.acc_dir, f"{acc}.refwin_te.tsv.gz")
        if not os.path.exists(path):
            warn(f"no projected TE for {acc}")
            continue
        d = pd.read_csv(path, sep="\t")
        d = d[d["ORTHO_BP"] > 0]
        d[acc] = d["TE_BP"] / d["ORTHO_BP"]
        te_rows[acc] = d.set_index("WINDOW")[[acc, "ORTHO_BP"]].rename(columns={"ORTHO_BP": f"__ortho_{acc}"})
        n_te += 1
    if te_rows:
        te = pd.concat(te_rows.values(), axis=1)
        frac_cols = [c for c in te.columns if not c.startswith("__ortho_")]
        ortho_cols = [c for c in te.columns if c.startswith("__ortho_")]
        out = pd.DataFrame(index=te.index)
        out["TE_FRAC_MEAN"] = te[frac_cols].mean(axis=1)
        out["TE_FRAC_SD"] = te[frac_cols].std(axis=1)
        out["ORTHO_BP_MEAN"] = te[ortho_cols].mean(axis=1)
        out["N_ACC_ORTHO"] = te[ortho_cols].notna().sum(axis=1)
        for p in pops:
            cols = [a for a in frac_cols if pop_of.get(a) == p]
            if cols:
                out[f"TE_FRAC_{p}"] = te[cols].mean(axis=1)
        out = out.reset_index().rename(columns={"index": "WINDOW"})
        out.to_csv(os.path.join(args.out_dir, "window_te_projection.tsv"), sep="\t",
                   index=False, float_format="%.6g", na_rep="NA")
        info(f"window_te_projection.tsv from {n_te} accessions ({len(out):,} windows)")
    else:
        warn("no TE projections found — window_te_projection.tsv not written")

    # ------------------------------------------------- gene synteny conservation
    ref_genes_path = os.path.join(args.ref_dir, "genes_ColPEK.bed")
    if os.path.exists(ref_genes_path):
        genes = pd.read_csv(ref_genes_path, sep="\t", header=None,
                            names=["CHROM", "START", "END", "GENE"], dtype={0: str})
        genes["CHROM"] = genes["CHROM"].astype(str)
        counts = {p: np.zeros(len(genes), dtype=np.int32) for p in pops}
        total = np.zeros(len(genes), dtype=np.int32)
        idx = {g: i for i, g in enumerate(genes["GENE"])}
        n_files = 0
        for acc in accs:
            path = os.path.join(args.acc_dir, f"{acc}.refgenes_syntenic.txt.gz")
            if not os.path.exists(path):
                warn(f"no reference-gene coverage for {acc}")
                continue
            n_files += 1
            p = pop_of.get(acc)
            with gzip.open(path, "rt") as fh:
                for line in fh:
                    i = idx.get(line.strip())
                    if i is not None:
                        total[i] += 1
                        if p in counts:
                            counts[p][i] += 1
        genes["N_ACC_SYNTENIC"] = total
        genes["FREQ_SYNTENIC"] = total / max(n_files, 1)
        for p in pops:
            n_p = sum(1 for a in accs if pop_of.get(a) == p)
            genes[f"FREQ_SYN_{p}"] = counts[p] / max(n_p, 1)
        f = genes["FREQ_SYNTENIC"]
        genes["SYN_CLASS"] = np.where(f >= args.core_frac, "syn_core",
                             np.where(f >= args.shell_frac, "syn_shell", "syn_cloud"))
        if args.pancore and os.path.exists(args.pancore):
            pc = pd.read_csv(args.pancore, sep="\t", header=None, names=["GENE", "PANCORE_CLASS"])
            genes = genes.merge(pc, on="GENE", how="left")
            info("joined the external pan-core classification")
        genes.to_csv(os.path.join(args.out_dir, "genes_ColPEK_synteny.tsv"), sep="\t",
                     index=False, float_format="%.6g", na_rep="NA")
        info(f"genes_ColPEK_synteny.tsv: {len(genes):,} genes from {n_files} accessions")
        info("  " + genes["SYN_CLASS"].value_counts().to_string().replace("\n", "; "))

        # per window
        mid = ((genes["START"] + genes["END"]) // 2).to_numpy()
        wid = [f"{c}_{((m) // W) * W + 1}" for c, m in zip(genes["CHROM"], mid)]
        genes["_W"] = wid
        agg = genes.groupby("_W").agg(
            GENE_COUNT=("GENE", "size"),
            GENE_SYNFREQ_MEAN=("FREQ_SYNTENIC", "mean"),
            GENE_SYNFREQ_MIN=("FREQ_SYNTENIC", "min"),
        )
        for cls in ["syn_core", "syn_shell", "syn_cloud"]:
            agg[f"GENE_{cls.upper()}"] = genes[genes["SYN_CLASS"] == cls].groupby("_W").size()
        agg = agg.fillna(0).reset_index().rename(columns={"_W": "WINDOW"})
        agg.to_csv(os.path.join(args.out_dir, "window_genes.tsv"), sep="\t",
                   index=False, float_format="%.6g")
        info(f"window_genes.tsv: {len(agg):,} windows with genes")
    else:
        warn(f"no {ref_genes_path} — gene layer skipped")

    # ------------------------------------------------------------------ SVs
    sv_path = os.path.join(args.ref_dir, "sv_sites.bed")
    if os.path.exists(sv_path):
        sv = pd.read_csv(sv_path, sep="\t")
        sv = sv.rename(columns={sv.columns[0]: "CHROM"})
        sv["CHROM"] = sv["CHROM"].astype(str)
        cnt, bp, af_sum = {}, {}, {}
        pop_cnt = {p: {} for p in pops}
        for row in sv.itertuples(index=False):
            wid = window_id(row.CHROM, row.START + 1, W)
            cnt[wid] = cnt.get(wid, 0) + 1
            bp[wid] = bp.get(wid, 0) + abs(int(row.SVLEN))
            af_sum[wid] = af_sum.get(wid, 0.0) + float(row.AF)
            for p in pops:
                col = f"AF_{p}"
                if col in sv.columns and getattr(row, col, 0) > 0:
                    pop_cnt[p][wid] = pop_cnt[p].get(wid, 0) + 1
        wins = sorted(cnt)
        out = pd.DataFrame({"WINDOW": wins,
                            "SV_COUNT": [cnt[w] for w in wins],
                            "SV_BP": [bp[w] for w in wins],
                            "SV_AF_MEAN": [af_sum[w] / cnt[w] for w in wins]})
        for p in pops:
            out[f"SV_COUNT_{p}"] = [pop_cnt[p].get(w, 0) for w in wins]
        out.to_csv(os.path.join(args.out_dir, "window_sv.tsv"), sep="\t", index=False, float_format="%.6g")
        info(f"window_sv.tsv: {len(sv):,} SVs in {len(out):,} windows")
    else:
        warn(f"no {sv_path} — SV layer skipped")

    # --------------------------------------------------------- centromeres
    cen_path = os.path.join(args.ref_dir, "centromeres.bed")
    if os.path.exists(cen_path) and os.path.getsize(cen_path) > 0:
        acc_bp = {}
        with open(cen_path) as fh:
            for line in fh:
                f = line.split("\t")
                if len(f) >= 3:
                    spread_interval(acc_bp, f[0], int(f[1]), int(f[2]), W)
        out = pd.DataFrame({"WINDOW": list(acc_bp), "CEN_BP": list(acc_bp.values())})
        out["CEN_FRAC"] = (out["CEN_BP"] / W).clip(0, 1)
        out.to_csv(os.path.join(args.out_dir, "window_cen.tsv"), sep="\t", index=False, float_format="%.6g")
        info(f"window_cen.tsv: {len(out):,} windows touching a centromere")

    # ---------------------------------------------- accession-level TE tables
    te_files = sorted(glob.glob(os.path.join(args.acc_dir, "*.te_synteny.tsv")))
    if te_files:
        allte = pd.concat([pd.read_csv(f, sep="\t") for f in te_files], ignore_index=True)
        allte.to_csv(os.path.join(args.out_dir, "te_synteny_per_accession.tsv"), sep="\t",
                     index=False, float_format="%.6g")
        sub = allte[allte["classification"] == "ALL"].copy()
        sub["te_enrichment_nonsyntenic"] = sub["te_frac_nonsyntenic"] / sub["te_frac_syntenic"].replace(0, np.nan)
        per_pop = sub.groupby("pop").agg(
            n=("accession", "size"),
            te_frac_syntenic=("te_frac_syntenic", "mean"),
            te_frac_nonsyntenic=("te_frac_nonsyntenic", "mean"),
            te_enrichment_nonsyntenic=("te_enrichment_nonsyntenic", "mean"),
        ).reset_index()
        per_pop.to_csv(os.path.join(args.out_dir, "te_synteny_per_pop.tsv"), sep="\t",
                       index=False, float_format="%.6g")
        info(f"te_synteny_per_accession.tsv: {len(te_files)} accessions")
        info("  TE fraction in syntenic vs non-syntenic sequence, per population:")
        print(per_pop.to_string(index=False), file=sys.stderr)
    else:
        sub = pd.DataFrame()

    # --------------------------------------------------- individual summary
    ind = pd.DataFrame({"accession": accs})
    ind["pop"] = ind["accession"].map(pop_of)
    if args.syn_per_accession and os.path.exists(args.syn_per_accession):
        s = pd.read_csv(args.syn_per_accession, sep="\t")[["accession", "frac_syntenic_ref", "n_syntenic_blocks"]]
        ind = ind.merge(s, on="accession", how="left")
    if len(sub):
        ind = ind.merge(sub[["accession", "te_frac_syntenic", "te_frac_nonsyntenic",
                             "te_enrichment_nonsyntenic"]], on="accession", how="left")
    gene_files = sorted(glob.glob(os.path.join(args.acc_dir, "*.gene_synteny.tsv")))
    if gene_files:
        g = pd.concat([pd.read_csv(f, sep="\t") for f in gene_files], ignore_index=True)
        ind = ind.merge(g[["accession", "n_genes", "n_genes_syntenic", "frac_syntenic"]]
                        .rename(columns={"frac_syntenic": "frac_own_genes_syntenic"}),
                        on="accession", how="left")
    if args.sv_per_accession and os.path.exists(args.sv_per_accession):
        sv = pd.read_csv(args.sv_per_accession, sep="\t")[["accession", "n_sv_carried", "n_ins", "n_del", "sv_bp_carried"]]
        ind = ind.merge(sv, on="accession", how="left")
    ind.to_csv(os.path.join(args.out_dir, "individual_summary.tsv"), sep="\t",
               index=False, float_format="%.6g", na_rep="NA")
    info(f"individual_summary.tsv: {ind.shape[0]} accessions x {ind.shape[1]} columns")

    # a couple of headline correlations at the individual level
    for x, y in [("frac_syntenic_ref", "n_sv_carried"),
                 ("frac_syntenic_ref", "te_frac_nonsyntenic"),
                 ("frac_syntenic_ref", "te_enrichment_nonsyntenic")]:
        if x in ind.columns and y in ind.columns:
            d = ind[[x, y]].dropna()
            if len(d) > 5:
                r = float(np.corrcoef(d[x], d[y])[0, 1])
                info(f"  individual level: corr({x}, {y}) = {r:.3f}  (n = {len(d)})")


if __name__ == "__main__":
    main()
