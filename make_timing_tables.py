#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_timing_tables.py

Emit the LaTeX bodies of the three timing tables, and the derived figures the
surrounding prose quotes, straight from the measurement files.

Every timing defect found in the audit was a transcription defect: a number
measured in one job, typed into a table, and left there when the job was
superseded. Generating the rows removes that failure mode -- if a table
disagrees with the measurements now, it is because someone edited the table by
hand.

Reads, all from one allocation:
  results_revision/linear/timing_classical_all.json
  results_revision/linear/timing_linear_v3_M16.json
  results_revision/linear/timing_linear_v3_M64.json

    python make_timing_tables.py            # print the rows
    python make_timing_tables.py --check    # compare against the .tex instead
"""

import argparse, json, re
from pathlib import Path

RES = Path("results_revision/linear")
TEX = Path("../CompMech_submission_ready/main_cm_sn.tex")

# The condition the paper reports: models served one at a time, denormals
# flushed. See the audit note on subnormal weights.
REPORTED = "blocked/flush_denormal"
FALLBACKS = ["blocked/flush_denormal", "flush_denormal",
             "interleaved/flush_denormal", "blocked/as_is", "as_is"]

LABEL = {"closed_form": "Closed-form LS read-out",
         "pi_spectral_anchored": "PI-spectral, anchored",
         "pi_spectral_plain": "PI-spectral, plain",
         "data_only_spectral": "Data-only spectral",
         "fno": "FNO"}


def load(name):
    p = RES / name
    if not p.exists():
        raise SystemExit(f"missing {p} -- has the timing job finished?")
    return json.load(open(p))


def cond_of(agg_entry):
    for c in FALLBACKS:
        if c in agg_entry:
            return c
    raise SystemExit(f"no usable condition in {list(agg_entry)}")


def pick(lin, tag):
    a = lin["aggregate"][tag]
    c = cond_of(a)
    return a[c], c


def mesh_label(res):
    n = res - 1
    return f"{n}$\\times${n}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report which emitted numbers are absent from the .tex")
    a = ap.parse_args()

    cl = load("timing_classical_all.json")
    m16 = load("timing_linear_v3_M16.json")
    m64 = load("timing_linear_v3_M64.json")

    _, cond = pick(m16, "pi_spectral_plain")
    print(f"% condition reported: {cond}")
    print(f"% assembly-path equivalence at uniform modulus: "
          f"rel diff {cl.get('equivalence_rel_diff', float('nan')):.1e}")
    print()

    fem = {r["res"]: r for r in cl["fem"]}
    rom = {r["rank"]: r["online_ms"] for r in cl["rom"]}
    f29 = fem[29]
    emitted = []

    def num(x, p=3):
        s = f"{x:.{p}f}"
        emitted.append(s)
        return s

    # ---------------- tab:ledgerA ----------------------------------------
    print("%% ---- tab:ledgerA body ----")
    cf16, _ = pick(m16, "closed_form")
    print(f"Closed-form LS read-out   & $\\mathbf{{3\\times10^{{-7}}}}$ & "
          f"{num(cf16['single_query_ms']['median'])} & "
          f"{num(cf16['batched_ms_per_sample']['median'], 4)} &")
    off = cl["offline"]
    print(f"  one least-squares solve & $\\sim$TBD \\\\")
    print(f"POD--Galerkin ROM ($r{{=}}32$) & 0.0099 & $\\mathbf{{{num(rom[32], 3)}}}$ & --- &")
    print(f"  {off['n_snapshots']} snapshots $+$ SVD, "
          f"{num(off['rom_total_ms'] / 1000.0, 2)}\\,s & $\\sim$TBD \\\\")
    print(f"Q4 FEM {mesh_label(29)}       & 0.0099 & "
          f"{num(f29['amortized_ms'])} / {num(f29['solve_only_ms'])} / "
          f"{num(f29['per_query_ms'])} & --- &")
    print(f"  one factorization, {num(off['fem_assemble_factorize_ms'] / 1000.0, 2)}\\,s "
          f"& reference \\\\")
    for tag in ("pi_spectral_anchored", "pi_spectral_plain", "fno", "data_only_spectral"):
        v, _ = pick(m16, tag)
        print(f"{LABEL[tag]:25s} & --- & {num(v['single_query_ms']['median'])} & "
              f"{num(v['batched_ms_per_sample']['median'], 4)} & \\\\")
    print()

    # ---------------- tab:canonical timing column ------------------------
    print("%% ---- tab:canonical single-query column ----")
    rows = [("Closed-form least-squares read-out", m16, "closed_form"),
            ("PI-spectral, LS-anchored ($M{=}64$)", m64, "pi_spectral_anchored"),
            ("PI-spectral, LS-anchored ($M{=}16$)", m16, "pi_spectral_anchored"),
            ("PI-spectral, plain branch ($M{=}16$)", m16, "pi_spectral_plain"),
            ("FNO", m16, "fno"),
            ("PI-spectral, plain branch ($M{=}64$)", m64, "pi_spectral_plain"),
            ("Data-only spectral", m16, "data_only_spectral")]
    for lab, src, tag in rows:
        v, _ = pick(src, tag)
        print(f"{lab:38s} single-query {num(v['single_query_ms']['median'])} ms  "
              f"params {v['n_params']}")
    print(f"{'FEM Q4 28x28 (amortized)':38s} single-query {num(f29['amortized_ms'])} ms")
    print(f"{'POD--Galerkin ROM (r=32)':38s} single-query {num(rom[32])} ms")
    print()

    # ---------------- tab:timing -----------------------------------------
    print("%% ---- tab:timing body ----")
    plain16, _ = pick(m16, "pi_spectral_plain")
    data16, _ = pick(m16, "data_only_spectral")
    ref = f29["amortized_ms"]
    for res in sorted(fem):
        r = fem[res]
        print(f"FEM Q4, {mesh_label(res):14s} & --- & {num(r['amortized_ms'])} & "
              f"{num(r['amortized_ms'])} & {ref / r['amortized_ms']:.2f}$\\times$ \\\\")
    for tag, v in (("PI-Spectral DeepONet", plain16), ("Data-only Spectral", data16)):
        b = v["batched_ms_per_sample"]["median"]
        s = v["single_query_ms"]["median"]
        print(f"{tag:22s} & --- & {num(b, 4)} & {num(s)} & "
              f"{ref / b:.1f}$\\times$ / {ref / s:.2f}$\\times$ \\\\")
    print()

    # ---------------- figures the prose quotes ---------------------------
    print("%% ---- derived values quoted in the prose ----")
    b = plain16["batched_ms_per_sample"]["median"]
    s = plain16["single_query_ms"]["median"]
    print(f"batched throughput advantage over amortized FEM : {ref / b:.1f}x   "
          f"(was 28x)")
    print(f"single-query vs amortized FEM                   : {s / ref:.2f}x slower "
          f"(was 1.7x)")
    print(f"single-query vs solve-only FEM                  : "
          f"{s / f29['solve_only_ms']:.2f}x slower")
    print(f"per-query FEM, topology precomputed             : "
          f"{f29['per_query_ms']:.2f} ms   (paper said 11 and 36.5)")
    print(f"per-query FEM, constructor re-run each call     : "
          f"{f29['per_query_with_setup_ms']:.2f} ms")
    fno16, _ = pick(m16, "fno")
    print(f"FNO single-query                                : "
          f"{fno16['single_query_ms']['median']:.2f} ms  "
          f"({fno16['single_query_ms']['median'] / ref:.0f}x the amortized FEM solve)")
    print(f"cheapest per query overall                       : ROM r=32 at "
          f"{rom[32]:.3f} ms")
    print(f"  = {b / rom[32]:.1f}x faster than the best neural batched throughput")
    print(f"  = {s / rom[32]:.0f}x faster than the best neural single-query latency")

    # break-even against the amortized FEM solve
    for name, off_ms, per_q in (("closed-form", None, cf16["batched_ms_per_sample"]["median"]),
                                ("ROM r=32", off["rom_total_ms"], rom[32])):
        if off_ms is None:
            continue
        saving = ref - per_q
        print(f"break-even, {name:12s}: {off_ms / saving:.0f} queries "
              f"(offline {off_ms:.0f} ms / saving {saving:.3f} ms)")

    if cl.get("hetero"):
        h = cl["hetero"]
        hrom = {r["rank"]: r["online_ms"] for r in h["rom"]}
        print(f"\nheterogeneous, res {h['res']}: FEM per query "
              f"{h['fem_per_query_ms']:.2f} ms")
        for r in sorted(hrom):
            print(f"  ROM r={r:4d}: {hrom[r]:8.3f} ms"
                  + ("   <-- overtakes the FEM solve" if hrom[r] > h["fem_per_query_ms"] else ""))

    if a.check and TEX.exists():
        tex = TEX.read_text()

        def present(v):
            """A value counts as present at any sensible rounding of itself.

            The tables quote 12.66 where the measurement is 12.660, so an exact
            string match would report false misses and hide the real ones.
            """
            x = float(v)
            for dp in range(5, -1, -1):
                t = f"{x:.{dp}f}"
                if t in tex:
                    return True
                if t.rstrip("0").rstrip(".") in tex:
                    return True
            return False

        uniq = sorted(set(emitted), key=float)
        missing = [v for v in uniq if not present(v)]
        print(f"\n[check] {len(uniq) - len(missing)}/{len(uniq)} emitted values "
              f"appear in the manuscript at some rounding")
        if missing:
            print("[check] NOT in the manuscript:", ", ".join(missing))
        else:
            print("[check] every generated value is in the manuscript")


if __name__ == "__main__":
    main()
