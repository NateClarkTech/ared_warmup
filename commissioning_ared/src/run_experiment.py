"""Run one commissioning-then-stream experiment.

Example:
    python -m src.run_experiment --method farpoint_then_ared \\
        --data PARKING_LOT_DINO --t-frac 0.1 --B0 50 --kappa 0.5 \\
        --query-rule global_fft --seed 0 --out results/run.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from src.farpoint_fft import QUERY_RULES
from src.plotting import plot_all, read_jsonl, write_summary_csv
from src.protocols import METHODS, run_configured


def _json_default(value: Any):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Not JSON serializable: {type(value)!r}")


def write_record(record: Dict[str, Any], path: Path, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "a"
    with path.open(mode, encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=_json_default) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Commission a pool, then stream the rest with A/RED.")
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--data", default="NICE", help="NICE, PARKING_LOT_DINO, EMNIST_DINO, or MNIST_DINO")
    parser.add_argument("--t-frac", type=float, default=0.2, dest="t_frac")
    parser.add_argument("--B0", type=int, default=20)
    parser.add_argument("--kappa", type=float, default=None, help="Default 0.5 on parking-lot DINO, 1.0 on NICE")
    parser.add_argument("--query-rule", default="global_fft", choices=QUERY_RULES, dest="query_rule")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/run.jsonl")
    parser.add_argument("--stream", default="stationary", choices=["stationary", "late_arrival"])
    parser.add_argument("--l-buf-size", type=int, default=1000, dest="l_buf_size")
    parser.add_argument("--n-rel-classes", type=int, default=None, dest="n_rel_classes")
    parser.add_argument(
        "--nice-base",
        type=int,
        default=800,
        dest="nice_base",
        help="Scaled NICE base count. 0 loads the full vendored generator (~1e6 points).",
    )
    parser.add_argument("--nice-min-count", type=int, default=20, dest="nice_min_count")
    parser.add_argument("--qs-var", type=int, default=1, choices=[0, 1], dest="qs_var")
    parser.add_argument("--k-comp-pts", type=int, default=2, dest="k_comp_pts")
    parser.add_argument(
        "--first-query",
        default="farthest_centroid",
        choices=["farthest_centroid", "random"],
        dest="first_query",
    )
    parser.add_argument("--smart-forgetting", type=int, default=3, dest="smart_forgetting_var")
    parser.add_argument(
        "--smart-forgetting-threshold",
        type=float,
        default=0.01,
        dest="smart_forgetting_threshold",
    )
    parser.add_argument("--neighborhood-merge", action="store_true", dest="neighborhood_merge")
    parser.add_argument("--overwrite", action="store_true", help="Replace the output JSONL instead of appending")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--plot", action="store_true", help="Redraw figures from the output JSONL after the run")
    parser.add_argument("--fig-dir", default="results/figures", dest="fig_dir")
    parser.add_argument("--summary", default="results/summary.csv")
    return parser


def cfg_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "method": args.method,
        "data": args.data,
        "t_frac": args.t_frac,
        "B0": args.B0,
        "kappa": args.kappa,
        "query_rule": args.query_rule,
        "seed": args.seed,
        "stream": args.stream,
        "l_buf_size": args.l_buf_size,
        "n_rel_classes": args.n_rel_classes,
        "nice_base": args.nice_base,
        "nice_min_count": args.nice_min_count,
        "qs_var": args.qs_var,
        "k_comp_pts": args.k_comp_pts,
        "first_query": args.first_query,
        "smart_forgetting": (args.smart_forgetting_var, args.smart_forgetting_threshold),
        "neighborhood_merge": bool(args.neighborhood_merge),
        "verbose": not args.quiet,
        "progress": not args.quiet,
    }


def main(argv: Optional[list] = None) -> Dict[str, Any]:
    args = build_parser().parse_args(argv)
    record = run_configured(cfg_from_args(args))
    out = Path(args.out)
    write_record(record, out, overwrite=bool(args.overwrite))
    print(f"wrote {out}")
    if args.plot:
        records = read_jsonl(out)
        write_summary_csv(records, Path(args.summary))
        plot_all(records, Path(args.fig_dir))
    return record


if __name__ == "__main__":
    main()
