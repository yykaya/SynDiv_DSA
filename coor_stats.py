#!/usr/bin/env python3
"""
coor_stats.py — individual-level synteny from a SynDiv `coor.out`.

One streaming pass over the 77-accession syntenic-coordinate table gives every
per-accession product we need downstream:

  per_accession_synteny.tsv      genome-wide syntenic bp / fraction per accession
  per_accession_chr_synteny.tsv  the same, split by chromosome
  beds/<ACC>.syntenic.bed.gz     merged syntenic blocks in reference coordinates
  synteny_frac_w<W>.tsv.gz       window x accession matrix, fraction syntenic
  synteny_freq_w<W>.tsv.gz       per-window synteny frequency, overall and per population

coor.out layout (SynDiv_c coor):
  chrom  refStart  refEnd  name1 qs1 qe1  name2 qs2 qe2 ...
Coordinates are 1-based inclusive; `0 0` marks an accession as non-syntenic in
that reference interval.
"""

from __future__ import annotations

import argparse
import gzip
import itertools
import os
import sys
from collections import OrderedDict

import numpy as np


def opener(path, mode="rt"):
    if str(path).endswith((".gz", ".GZ")):
        return gzip.open(path, mode)
    return open(path, mode)


def read_chrom_sizes(fai_path):
    sizes = OrderedDict()
    with open(fai_path) as fh:
        for line in fh:
            if not line.strip():
                continue
            f = line.split("\t")
            sizes[f[0]] = int(f[1])
    return sizes


def read_pop_map(path):
    pops = {}
    if not path:
        return pops
    with open(path) as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) >= 2 and f[0]:
                pops[f[0]] = f[1]
    return pops


class BedWriter:
    """Lazily merges contiguous syntenic reference intervals per accession."""

    def __init__(self, outdir, names, enabled=True):
        self.enabled = enabled
        self.names = names
        self.n = len(names)
        self.cur_chrom = None
        self.start = np.zeros(self.n, dtype=np.int64)
        self.end = np.zeros(self.n, dtype=np.int64)   # 0 = no open block
        self.blocks = np.zeros(self.n, dtype=np.int64)
        self.fh = []
        if enabled:
            os.makedirs(outdir, exist_ok=True)
            for name in names:
                self.fh.append(gzip.open(os.path.join(outdir, f"{name}.syntenic.bed.gz"), "wt", compresslevel=1))

    def _flush(self, idx):
        if not self.enabled:
            return
        chrom = self.cur_chrom
        for i in idx:
            # BED is 0-based half-open; coor.out is 1-based inclusive
            self.fh[i].write(f"{chrom}\t{self.start[i] - 1}\t{self.end[i]}\n")

    def add(self, chrom, s, e, syn_idx):
        if chrom != self.cur_chrom:
            self.close_open()
            self.cur_chrom = chrom
        if syn_idx.size == 0:
            return
        open_here = self.end[syn_idx] > 0
        contiguous = np.zeros(syn_idx.size, dtype=bool)
        contiguous[open_here] = self.end[syn_idx[open_here]] == s - 1

        extend = syn_idx[contiguous]
        self.end[extend] = e

        fresh = syn_idx[~contiguous]
        if fresh.size:
            to_flush = fresh[self.end[fresh] > 0]
            self._flush(to_flush)
            self.start[fresh] = s
            self.end[fresh] = e
            self.blocks[fresh] += 1

    def close_open(self):
        if self.cur_chrom is None:
            return
        idx = np.flatnonzero(self.end > 0)
        self._flush(idx)
        self.end[:] = 0
        self.start[:] = 0

    def close(self):
        self.close_open()
        for fh in self.fh:
            fh.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coor", required=True, help="SynDiv coor.out (plain or .gz)")
    ap.add_argument("--fai", required=True, help="reference .fai (chrom <TAB> length ...)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--pop-map", default="", help="accession <TAB> population")
    ap.add_argument("--pops", default="", help="comma-separated population order for the frequency table")
    ap.add_argument("--windows", default="5000,100000", help="comma-separated window sizes (non-overlapping)")
    ap.add_argument("--present-frac", type=float, default=0.5,
                    help="a window counts as syntenic for an accession when this fraction is covered [0.5]")
    ap.add_argument("--no-beds", action="store_true", help="skip the per-accession BED files")
    ap.add_argument("--chroms", default="", help="comma-separated chromosomes to keep (default: all in the .fai)")
    ap.add_argument("--progress", type=int, default=2_000_000, help="log every N lines")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    sizes = read_chrom_sizes(args.fai)
    if args.chroms:
        keep = [c for c in args.chroms.split(",") if c]
        sizes = OrderedDict((c, sizes[c]) for c in keep if c in sizes)
    if not sizes:
        sys.exit("no chromosomes selected — check --fai / --chroms")

    windows = [int(w) for w in args.windows.split(",") if w]
    pop_of = read_pop_map(args.pop_map)
    pop_order = [p for p in args.pops.split(",") if p] or sorted(set(pop_of.values()))

    fh = opener(args.coor)
    first = fh.readline()
    if not first:
        sys.exit(f"empty coor file: {args.coor}")
    fields = first.rstrip("\n").split("\t")
    if len(fields) < 6 or (len(fields) - 3) % 3 != 0:
        sys.exit(f"unexpected coor.out layout: {len(fields)} columns")
    names = fields[3::3]
    n_acc = len(names)
    print(f"[coor_stats] {n_acc} accessions, {len(sizes)} chromosomes", file=sys.stderr)

    # ---------------------------------------------------------- accumulators
    syn_bp = np.zeros(n_acc, dtype=np.int64)
    syn_bp_chr = {c: np.zeros(n_acc, dtype=np.int64) for c in sizes}
    covered_chr = {c: 0 for c in sizes}
    win_bp = {}       # window size -> chrom -> (n_acc x n_win) syntenic bp
    for w in windows:
        win_bp[w] = {c: np.zeros((n_acc, (L + w - 1) // w), dtype=np.int32) for c, L in sizes.items()}

    beds = BedWriter(os.path.join(args.outdir, "beds"), names, enabled=not args.no_beds)

    # --------------------------------------------------------------- streaming
    n_rows = 0
    n_skipped = 0
    for line in itertools.chain([first], fh):
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 6:
            continue
        chrom = parts[0]
        if chrom not in sizes:
            n_skipped += 1
            continue
        s = int(parts[1])
        e = int(parts[2])
        if e < s:
            continue

        length = e - s + 1
        qs = parts[4::3]
        qe = parts[5::3]
        syn = np.fromiter(
            (0 if (a == "0" and b == "0") else 1 for a, b in zip(qs, qe)),
            dtype=np.int8, count=n_acc,
        )
        idx = np.flatnonzero(syn)

        covered_chr[chrom] += length
        if idx.size:
            syn_bp[idx] += length
            syn_bp_chr[chrom][idx] += length
            for w in windows:
                arr = win_bp[w][chrom]
                w0 = (s - 1) // w
                w1 = (e - 1) // w
                if w0 == w1:
                    arr[idx, w0] += length
                else:
                    arr[idx, w0] += (w0 + 1) * w - (s - 1)
                    if w1 > w0 + 1:
                        arr[np.ix_(idx, np.arange(w0 + 1, w1))] += w
                    arr[idx, w1] += e - w1 * w
        beds.add(chrom, s, e, idx)

        n_rows += 1
        if args.progress and n_rows % args.progress == 0:
            print(f"[coor_stats] {n_rows:,} rows", file=sys.stderr)

    fh.close()
    beds.close()
    print(f"[coor_stats] {n_rows:,} rows used, {n_skipped:,} skipped (chromosome not in .fai)", file=sys.stderr)

    # -------------------------------------------------- per-accession tables
    ref_total = sum(sizes.values())
    with open(os.path.join(args.outdir, "per_accession_synteny.tsv"), "w") as out:
        out.write("accession\tpop\tsyntenic_bp\tnonsyntenic_bp\tref_bp\tcovered_bp\t"
                  "frac_syntenic_ref\tfrac_syntenic_covered\tn_syntenic_blocks\n")
        covered_total = sum(covered_chr.values())
        order = np.argsort(-syn_bp)
        for i in order:
            pop = pop_of.get(names[i], "NA")
            frac_ref = syn_bp[i] / ref_total if ref_total else 0.0
            frac_cov = syn_bp[i] / covered_total if covered_total else 0.0
            out.write(f"{names[i]}\t{pop}\t{syn_bp[i]}\t{ref_total - syn_bp[i]}\t{ref_total}\t{covered_total}\t"
                      f"{frac_ref:.6f}\t{frac_cov:.6f}\t{int(beds.blocks[i])}\n")

    with open(os.path.join(args.outdir, "per_accession_chr_synteny.tsv"), "w") as out:
        out.write("accession\tpop\tchrom\tsyntenic_bp\tchrom_bp\tcovered_bp\tfrac_syntenic\n")
        for c, L in sizes.items():
            for i in range(n_acc):
                pop = pop_of.get(names[i], "NA")
                out.write(f"{names[i]}\t{pop}\t{c}\t{syn_bp_chr[c][i]}\t{L}\t{covered_chr[c]}\t"
                          f"{syn_bp_chr[c][i] / L if L else 0:.6f}\n")

    # ------------------------------------------------------- window products
    pop_idx = {p: np.array([i for i, nm in enumerate(names) if pop_of.get(nm) == p], dtype=np.int64)
               for p in pop_order}

    for w in windows:
        tag = f"w{w // 1000}k" if w % 1000 == 0 else f"w{w}"
        frac_path = os.path.join(args.outdir, f"synteny_frac_{tag}.tsv.gz")
        freq_path = os.path.join(args.outdir, f"synteny_freq_{tag}.tsv.gz")

        with gzip.open(frac_path, "wt", compresslevel=6) as fout, \
             gzip.open(freq_path, "wt", compresslevel=6) as qout:
            fout.write("#CHROM\tSTART\tEND\t" + "\t".join(names) + "\n")
            header = ["#CHROM", "START", "END", "WIN_BP", "N_ACC", "MEAN_FRAC_ALL", "N_PRESENT_ALL", "FREQ_ALL"]
            for p in pop_order:
                header += [f"MEAN_FRAC_{p}", f"FREQ_{p}"]
            qout.write("\t".join(header) + "\n")

            for c, L in sizes.items():
                arr = win_bp[w][c]                      # n_acc x n_win, syntenic bp
                n_win = arr.shape[1]
                starts = np.arange(n_win, dtype=np.int64) * w + 1
                ends = np.minimum(starts + w - 1, L)
                lens = (ends - starts + 1).astype(np.float64)
                frac = np.clip(arr / lens[None, :], 0.0, 1.0)
                present = frac >= args.present_frac

                mean_all = frac.mean(axis=0)
                n_present = present.sum(axis=0)
                pop_stats = []
                for p in pop_order:
                    ii = pop_idx[p]
                    if ii.size:
                        pop_stats.append((frac[ii].mean(axis=0), present[ii].mean(axis=0)))
                    else:
                        pop_stats.append((np.zeros(n_win), np.zeros(n_win)))

                for j in range(n_win):
                    fout.write(f"{c}\t{starts[j]}\t{ends[j]}\t" +
                               "\t".join(f"{v:.4f}" for v in frac[:, j]) + "\n")
                    row = [c, str(starts[j]), str(ends[j]), str(int(lens[j])), str(n_acc),
                           f"{mean_all[j]:.4f}", str(int(n_present[j])), f"{n_present[j] / n_acc:.4f}"]
                    for mf, fq in pop_stats:
                        row += [f"{mf[j]:.4f}", f"{fq[j]:.4f}"]
                    qout.write("\t".join(row) + "\n")
        print(f"[coor_stats] wrote {frac_path}", file=sys.stderr)
        print(f"[coor_stats] wrote {freq_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
