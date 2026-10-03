"""Figures for the commissioning-then-stream comparison.

Curves are watch-phase metrics, split at T inside each run record. Methods are
compared at the same buffer and (except for the kappa sweep) the same kappa.
A dashed line marks the fraction of post-T relevant mass whose class already
appears in the pool. It is a reference, not an extra method, and watch-phase
recall can exceed it when a class shows up only after T.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

METHOD_ORDER = (
    "cold_ared",
    "random_then_ared",
    "farpoint_then_ared",
    "oracle_then_ared",
)
PALETTE = {
    "cold_ared": "#0072B2",
    "random_then_ared": "#E69F00",
    "farpoint_then_ared": "#009E73",
    "oracle_then_ared": "#CC79A7",
}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def records_to_frame(records: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    flat = []
    for record in records:
        comm = record.get("commissioning") or {}
        watch = record.get("watch") or {}
        whole = record.get("whole") or {}
        latency = watch.get("discovery_latency") or {}
        flat.append(
            {
                "method": record.get("method"),
                "data": record.get("data"),
                "stream": record.get("stream"),
                "seed": record.get("seed"),
                "t_frac": record.get("t_frac"),
                "T": record.get("T"),
                "N": record.get("N"),
                "B0": record.get("B0"),
                "kappa": record.get("kappa"),
                "l_buf_size": record.get("l_buf_size"),
                "n_rel_classes": record.get("n_rel_classes"),
                "query_rule": record.get("query_rule"),
                "held_out_class": record.get("held_out_class"),
                "n_rel_classes_in_pool": record.get("n_rel_classes_in_pool"),
                "post_T_rel_mass_from_pool_classes": record.get("post_T_rel_mass_from_pool_classes"),
                "seconds": record.get("seconds"),
                "comm_queries": comm.get("queries_used"),
                "comm_n_classes": comm.get("n_classes_discovered"),
                "comm_n_relevant": comm.get("n_relevant_discovered"),
                "comm_classes": json.dumps(comm.get("classes_discovered")),
                "comm_relevant": json.dumps(comm.get("relevant_classes_discovered")),
                "watch_qp": watch.get("query_precision"),
                "watch_rr": watch.get("relevant_recall"),
                "watch_queries": watch.get("num_queries"),
                "watch_query_rate": watch.get("query_rate"),
                "watch_rel_queried": watch.get("relevant_queried"),
                "watch_rel_streamed": watch.get("relevant_streamed"),
                "missed_relevant_mass": watch.get("missed_relevant_mass"),
                "missed_relevant_fraction": watch.get("missed_relevant_fraction"),
                "queries_on_known_irrelevant": watch.get("queries_on_known_irrelevant"),
                "total_labels": whole.get("total_labels"),
                "relevant_examples_found": whole.get("relevant_examples_found"),
                "labels_per_relevant_found": whole.get("labels_per_relevant_found"),
                "discovery_latency": json.dumps(latency),
                "_latency": latency,
                "_rel_classes": record.get("rel_classes") or list(latency.keys()),
            }
        )
    return pd.DataFrame(flat)


def write_summary_csv(records: Sequence[Dict[str, Any]], path: Path) -> pd.DataFrame:
    frame = records_to_frame(records)
    public = frame.drop(columns=[c for c in frame.columns if c.startswith("_")], errors="ignore")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    public.to_csv(path, index=False)
    return frame


def _focus_kappa(frame: pd.DataFrame) -> float:
    kappas = sorted(float(k) for k in frame["kappa"].dropna().unique())
    if not kappas:
        return 0.5
    if len(kappas) == 1:
        return kappas[0]
    for preferred in (0.5, 1.0):
        for kappa in kappas:
            if abs(kappa - preferred) < 1e-9:
                return kappa
    return float(np.median(kappas))


def _focus_lbuf(frame: pd.DataFrame) -> Optional[int]:
    values = [int(v) for v in frame["l_buf_size"].dropna().unique()]
    if not values:
        return None
    return max(values)


def _subset_for_b0(frame: pd.DataFrame, stream: str) -> pd.DataFrame:
    sub = frame[frame["stream"] == stream].copy()
    if sub.empty:
        return sub
    kappa = _focus_kappa(sub)
    lbuf = _focus_lbuf(sub)
    sub = sub[np.isclose(sub["kappa"].astype(float), kappa)]
    if lbuf is not None:
        sub = sub[sub["l_buf_size"].astype(int) == lbuf]
    return sub


def _errorbar(frame: pd.DataFrame):
    if frame.empty:
        return None
    keys = ["method", "B0"]
    if "t_frac" in frame.columns and frame["t_frac"].nunique() > 1:
        keys.append("t_frac")
    counts = frame.groupby(keys, dropna=False).size()
    return "sd" if len(counts) and int(counts.max()) > 1 else None


def _coverage_line(ax, frame: pd.DataFrame) -> None:
    if "post_T_rel_mass_from_pool_classes" not in frame.columns:
        return
    values = frame["post_T_rel_mass_from_pool_classes"].dropna().astype(float)
    if values.empty:
        return
    level = float(values.mean())
    ax.axhline(
        level,
        color="#444444",
        linestyle="--",
        linewidth=1.0,
        label=f"pool-class coverage ({level:.2f})",
    )


def _line_rr(frame: pd.DataFrame, stream: str, path: Path, title: str) -> bool:
    sub = _subset_for_b0(frame, stream)
    if sub.empty:
        return False
    kappa = float(sub["kappa"].iloc[0])
    lbuf = int(sub["l_buf_size"].iloc[0])
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    # Dodge markers. On an easy stream every method can sit at the same recall,
    # and an undodged line hides all but the last method.
    plot_df = sub.copy()
    if plot_df["t_frac"].nunique() > 1:
        plot_df["B0_label"] = plot_df.apply(lambda row: f"{row['B0']:g}\nT={row['t_frac']:g}", axis=1)
    else:
        plot_df["B0_label"] = plot_df["B0"].map(lambda value: f"{value:g}")
    sns.pointplot(
        data=plot_df,
        x="B0_label",
        y="watch_rr",
        hue="method",
        hue_order=[m for m in METHOD_ORDER if m in set(plot_df["method"])],
        palette=PALETTE,
        dodge=0.35,
        linestyle="none",
        errorbar=_errorbar(plot_df),
        ax=ax,
    )
    _coverage_line(ax, sub)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel(f"Pool budget B0  (kappa={kappa:g}, buffer={lbuf})")
    ax.set_ylabel("Watch-phase relevant recall")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8, frameon=True)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


def _kappa_scatter(frame: pd.DataFrame, path: Path) -> bool:
    if frame.empty or frame["kappa"].nunique() < 1:
        return False
    sub = frame[frame["stream"] == "stationary"].copy()
    if sub.empty:
        sub = frame.copy()
    lbuf = _focus_lbuf(sub)
    if lbuf is not None:
        sub = sub[sub["l_buf_size"].astype(int) == lbuf]
    # One budget, one pool fraction, so the only sweep on the axes is kappa.
    if sub["B0"].nunique() > 1:
        b0 = float(np.median(sub["B0"].astype(float).unique()))
        # Snap to an actual budget value.
        choices = sorted(sub["B0"].astype(float).unique())
        b0 = min(choices, key=lambda value: abs(value - b0))
        sub = sub[np.isclose(sub["B0"].astype(float), b0)]
    else:
        b0 = float(sub["B0"].iloc[0]) if len(sub) else None
    if sub["t_frac"].nunique() > 1:
        t_frac = float(sorted(sub["t_frac"].unique())[0])
        sub = sub[np.isclose(sub["t_frac"].astype(float), t_frac)]
    else:
        t_frac = float(sub["t_frac"].iloc[0]) if len(sub) else None
    if sub.empty:
        return False
    grouped = (
        sub.groupby(["method", "kappa"], as_index=False)[["watch_qp", "watch_rr"]].mean()
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for method in METHOD_ORDER:
        piece = grouped[grouped["method"] == method].sort_values("kappa")
        if piece.empty:
            continue
        ax.plot(
            piece["watch_rr"],
            piece["watch_qp"],
            marker="o",
            color=PALETTE.get(method, "black"),
            label=method,
        )
        for _, row in piece.iterrows():
            ax.annotate(
                f"{row['kappa']:g}",
                (row["watch_rr"], row["watch_qp"]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
                color=PALETTE.get(method, "black"),
            )
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("Watch-phase relevant recall")
    ax.set_ylabel("Watch-phase query precision")
    ax.set_title("Query precision vs relevant recall across kappa")
    note = []
    if b0 is not None:
        note.append(f"B0={b0:g}")
    if lbuf is not None:
        note.append(f"buffer={lbuf}")
    if t_frac is not None:
        note.append(f"T fraction={t_frac:g}")
    if note:
        ax.text(0.01, 0.01, ", ".join(note), transform=ax.transAxes, fontsize=8, color="#333333")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


def _latency_frame(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    # to_dict keeps the leading-underscore columns. itertuples renames them.
    for record in frame.to_dict(orient="records"):
        latency = record.get("_latency") or {}
        classes = record.get("_rel_classes") or list(latency.keys())
        for label in classes:
            raw = latency.get(str(label), latency.get(label))
            rows.append(
                {
                    "method": record["method"],
                    "stream": record["stream"],
                    "seed": record["seed"],
                    "B0": record["B0"],
                    "kappa": record["kappa"],
                    "l_buf_size": record["l_buf_size"],
                    "t_frac": record["t_frac"],
                    "T": record["T"],
                    "N": record["N"],
                    "class": str(label),
                    "latency": np.nan if raw is None else float(raw),
                    "discovered": raw is not None,
                }
            )
    columns = [
        "method", "stream", "seed", "B0", "kappa", "l_buf_size", "t_frac",
        "T", "N", "class", "latency", "discovered",
    ]
    return pd.DataFrame(rows, columns=columns)


def _latency_plot(frame: pd.DataFrame, path: Path) -> bool:
    if frame.empty or "_latency" not in frame.columns:
        return False
    sub = frame.copy()
    kappa = _focus_kappa(sub)
    lbuf = _focus_lbuf(sub)
    sub = sub[np.isclose(sub["kappa"].astype(float), kappa)]
    if lbuf is not None:
        sub = sub[sub["l_buf_size"].astype(int) == int(lbuf)]
    if sub["B0"].nunique() > 1:
        choices = sorted(float(v) for v in sub["B0"].unique())
        target = float(np.median(choices))
        chosen = min(choices, key=lambda value: abs(value - target))
        sub = sub[np.isclose(sub["B0"].astype(float), chosen)]
    if sub["t_frac"].nunique() > 1:
        t_frac = float(sorted(sub["t_frac"].unique())[0])
        sub = sub[np.isclose(sub["t_frac"].astype(float), t_frac)]
    long = _latency_frame(sub)
    long = long.dropna(subset=["latency"])
    if long.empty:
        return False
    streams = [name for name in ("stationary", "late_arrival") if (long["stream"] == name).any()]
    if not streams:
        return False
    fig, axes = plt.subplots(1, len(streams), figsize=(6.2 * len(streams), 4.6), squeeze=False)
    for ax, stream in zip(axes[0], streams):
        piece = long[long["stream"] == stream]
        multi = int(piece.groupby(["method", "class"]).size().max()) > 1
        sns.barplot(
            data=piece,
            x="class",
            y="latency",
            hue="method",
            hue_order=[m for m in METHOD_ORDER if m in set(piece["method"])],
            palette=PALETTE,
            errorbar="sd" if multi else None,
            ax=ax,
        )
        if piece["T"].nunique() == 1:
            ax.axhline(float(piece["T"].iloc[0]), color="#444444", linestyle=":", linewidth=1.0, label="T")
        ax.set_xlabel("Relevant class")
        ax.set_ylabel("Stream index of first query")
        ax.set_title(f"Discovery latency ({stream})")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Discovery latency of relevant classes", fontsize=12)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


def plot_all(records: Sequence[Dict[str, Any]], out_dir: Path) -> List[Path]:
    """Write the four comparison figures. Missing slices are skipped."""
    sns.set_theme(style="whitegrid", context="notebook")
    frame = records_to_frame(records)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    targets = [
        (
            out_dir / "fig1_watch_rr_vs_b0_stationary.png",
            lambda: _line_rr(
                frame,
                "stationary",
                out_dir / "fig1_watch_rr_vs_b0_stationary.png",
                "Watch-phase relevant recall vs pool budget",
            ),
        ),
        (
            out_dir / "fig2_watch_rr_vs_b0_late_arrival.png",
            lambda: _line_rr(
                frame,
                "late_arrival",
                out_dir / "fig2_watch_rr_vs_b0_late_arrival.png",
                "Watch-phase relevant recall vs pool budget (late arrival)",
            ),
        ),
        (
            out_dir / "fig3_qp_vs_rr_kappa.png",
            lambda: _kappa_scatter(frame, out_dir / "fig3_qp_vs_rr_kappa.png"),
        ),
        (
            out_dir / "fig4_discovery_latency.png",
            lambda: _latency_plot(frame, out_dir / "fig4_discovery_latency.png"),
        ),
    ]
    for path, writer in targets:
        if writer():
            written.append(path)
            print(f"wrote {path}")
        else:
            print(f"skipped {path.name}: not enough rows")
    return written


def main(argv: Optional[Iterable[str]] = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Aggregate JSONL runs into a CSV and figures.")
    parser.add_argument("--jsonl", required=True, help="JSONL file or a directory of JSONL files")
    parser.add_argument("--out", default="results/figures", help="Figure directory")
    parser.add_argument("--summary", default="results/summary.csv")
    args = parser.parse_args(list(argv) if argv is not None else None)
    source = Path(args.jsonl)
    if source.is_dir():
        records = []
        for path in sorted(source.glob("*.jsonl")):
            records.extend(read_jsonl(path))
    else:
        records = read_jsonl(source)
    if not records:
        raise SystemExit(f"No records in {source}")
    write_summary_csv(records, Path(args.summary))
    print(f"wrote {args.summary} ({len(records)} rows)")
    plot_all(records, Path(args.out))


if __name__ == "__main__":
    main()
