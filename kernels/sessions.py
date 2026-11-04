"""Summarize saved runs: python -m kernels.sessions session1.json session2.json ..."""
import argparse
import json
from pathlib import Path
from statistics import median


def summarize(reports):
    if len({r["session"] for r in reports}) != len(reports):
        raise ValueError("session labels must be distinct")
    keys = ("gpu", "cc", "torch", "triton", "cuda", "warmup", "iterations", "repeats", "seed")
    groups = []
    for r in reports:
        if any(r[k] != reports[0][k] for k in keys):
            raise ValueError("hardware, versions and timing settings must match")
        groups.append({(v["kind"], v["n"], v["method"]): v for v in r["rows"]})
    if any(g.keys() != groups[0].keys() for g in groups):
        raise ValueError("benchmark cases differ across reports")
    def show(label, values, unit):
        print(f"{label}: {median(values):.3f} {unit} [{min(values):.3f}, {max(values):.3f}]")
    print(f"{len(groups)} sessions; median of session summaries [min, max]")
    for key in groups[0]:
        label = " ".join(map(str, key))
        show(label, [g[key]["median_us"] for g in groups], "us")
        show(label + " peak", [g[key]["peak_allocated_bytes"]/2**20 for g in groups], "MiB")
        kind, n, method = key
        if kind == "add" and method == "cuda":
            ratios = [median(a/b for a,b in zip(g[kind,n,"torch"]["samples_us"],
                                                g[key]["samples_us"], strict=True)) for g in groups]
            show(f"add {n} paired torch/cuda", ratios, "x")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reports", type=Path, nargs="+")
    paths = p.parse_args().reports
    if len(paths) < 2:
        p.error("provide reports from at least two independent GPU sessions")
    summarize([json.loads(path.read_text()) for path in paths])
