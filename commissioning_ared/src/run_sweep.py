"""Expand a YAML grid into JSONL rows, a summary CSV, and the four figures.

Cold start ignores B0, so each cold configuration is run once and copied onto
every budget. Every other method is run at each budget. The same preloaded
stream is reused across methods that share a seed.
"""

from __future__ import annotations

import argparse
import copy
import itertools
from pathlib import Path
from typing import Any, Dict, List

import yaml

from src.datasets import load_dataset
from src.plotting import plot_all, write_summary_csv
from src.run_experiment import write_record
from src.protocols import run_configured


def _as_list(value):
    if isinstance(value, list):
        return value
    return [value]


def _load_yaml(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a mapping")
    return loaded


def iter_runs(cfg: Dict[str, Any]):
    methods = [str(m) for m in _as_list(cfg.get("methods", ["farpoint_then_ared"]))]
    streams = [str(s) for s in _as_list(cfg.get("streams", ["stationary"]))]
    t_fracs = [float(t) for t in _as_list(cfg.get("t_fracs", [0.2]))]
    budgets = [int(b) for b in _as_list(cfg.get("B0s", [20]))]
    kappas = [float(k) for k in _as_list(cfg.get("kappas", [1.0]))]
    buffers = [int(b) for b in _as_list(cfg.get("l_buf_sizes", [1000]))]
    n_rels = [int(n) for n in _as_list(cfg.get("n_rel_classes", [4]))]
    seeds = [int(s) for s in _as_list(cfg.get("seeds", [0]))]
    for seed, stream, t_frac, kappa, l_buf, n_rel in itertools.product(
        seeds, streams, t_fracs, kappas, buffers, n_rels
    ):
        yield {
            "methods": methods,
            "budgets": budgets,
            "seed": seed,
            "stream": stream,
            "t_frac": t_frac,
            "kappa": kappa,
            "l_buf_size": l_buf,
            "n_rel_classes": n_rel,
        }


def _base_cfg(cfg: Dict[str, Any], bundle, job: Dict[str, Any], method: str, budget: int) -> Dict[str, Any]:
    smart = cfg.get("smart_forgetting", [3, 0.01])
    return {
        "method": method,
        "data": str(cfg.get("data", "NICE")),
        "nice_base": int(cfg.get("nice_base", 800)),
        "nice_min_count": int(cfg.get("nice_min_count", 20)),
        "query_rule": str(cfg.get("query_rule", "global_fft")),
        "qs_var": int(cfg.get("qs_var", 1)),
        "k_comp_pts": int(cfg.get("k_comp_pts", 2)),
        "first_query": str(cfg.get("first_query", "farthest_centroid")),
        "smart_forgetting": (int(smart[0]), float(smart[1])),
        "neighborhood_merge": bool(cfg.get("neighborhood_merge", False)),
        "verbose": bool(cfg.get("verbose", False)),
        "progress": bool(cfg.get("progress", False)),
        "bundle": bundle,
        "seed": job["seed"],
        "stream": job["stream"],
        "t_frac": job["t_frac"],
        "kappa": job["kappa"],
        "l_buf_size": job["l_buf_size"],
        "n_rel_classes": job["n_rel_classes"],
        "B0": int(budget),
    }


def run_sweep(cfg: Dict[str, Any], out_dir: Path) -> List[Dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "runs.jsonl"
    if jsonl_path.exists():
        jsonl_path.unlink()
    data_name = str(cfg.get("data", "NICE"))
    nice_base = int(cfg.get("nice_base", 800))
    nice_min = int(cfg.get("nice_min_count", 20))
    cache = {}
    records: List[Dict[str, Any]] = []
    jobs = list(iter_runs(cfg))
    total = sum(len(job["methods"]) * len(job["budgets"]) for job in jobs)
    done = 0
    for job in jobs:
        key = (job["seed"], job["n_rel_classes"], nice_base, nice_min, data_name)
        if key not in cache:
            cache[key] = load_dataset(
                data_name,
                n_rel_classes=job["n_rel_classes"],
                seed=job["seed"],
                nice_base=nice_base,
                nice_min_count=nice_min,
                verbose=False,
            )
        bundle = cache[key]
        for method in job["methods"]:
            budgets = job["budgets"]
            if method == "cold_ared":
                record = run_configured(_base_cfg(cfg, bundle, job, method, budgets[0]))
                for budget in budgets:
                    stamped = copy.deepcopy(record)
                    stamped["B0"] = int(budget)
                    stamped["B0_effective"] = 0
                    write_record(stamped, jsonl_path, overwrite=False)
                    records.append(stamped)
                    done += 1
                    _progress(done, total, stamped)
                continue
            for budget in budgets:
                record = run_configured(_base_cfg(cfg, bundle, job, method, budget))
                write_record(record, jsonl_path, overwrite=False)
                records.append(record)
                done += 1
                _progress(done, total, record)
    summary = out_dir / "summary.csv"
    write_summary_csv(records, summary)
    print(f"wrote {summary}")
    plot_all(records, out_dir / "figures")
    print(f"wrote {jsonl_path} ({len(records)} rows)")
    return records


def _progress(done: int, total: int, record: Dict[str, Any]) -> None:
    watch = record["watch"]
    comm = record["commissioning"]
    print(
        f"[{done}/{total}] {record['method']} stream={record['stream']} "
        f"seed={record['seed']} B0={record['B0']} kappa={record['kappa']} "
        f"T={record['T']}  "
        f"discovered={comm['relevant_classes_discovered']}  "
        f"QP={watch['query_precision']:.3f} RR={watch['relevant_recall']:.3f}"
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Run a YAML experiment matrix.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default=None, help="Output directory (default: results/<config stem>)")
    args = parser.parse_args(argv)
    cfg = _load_yaml(Path(args.config))
    out = Path(args.out) if args.out else Path(cfg.get("out", f"results/{Path(args.config).stem}"))
    run_sweep(cfg, out)


if __name__ == "__main__":
    main()
