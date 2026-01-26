#!/usr/bin/env python3
"""
Plot merged solution summaries with publication-quality bar charts.

Generates figures suitable for academic papers with clean aesthetics,
showing throughput and cost efficiency with TP/PP annotations.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

# Publication-quality color palette
COLORS = {
    "throughput": "#1f77b4",  # Matplotlib blue
    "cost": "#d62728",        # Matplotlib red
    "bar_edge": "#2c3e50",    # Dark slate
    "text": "#2c3e50",
    "grid": "#bdc3c7",
}


def setup_matplotlib_style() -> None:
    """Configure matplotlib for publication-quality output."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        # Font settings
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        # Figure settings
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        # Axes settings
        "axes.linewidth": 0.8,
        "axes.edgecolor": COLORS["text"],
        "axes.labelcolor": COLORS["text"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        # Grid settings
        "axes.grid": True,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.4,
        "grid.color": COLORS["grid"],
        # Tick settings
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.color": COLORS["text"],
        "ytick.color": COLORS["text"],
    })


def infer_workload(row: pd.Series) -> str:
    """Infer workload type from source path or model name."""
    source_path = str(row.get("source_path", "")).lower()
    model_name = str(row.get("model_name", "")).lower()
    if "wrk_prefill" in source_path:
        return "prefill"
    if "wrk_decode" in source_path:
        return "decode"
    if "prefill" in model_name:
        return "prefill"
    if "decode" in model_name:
        return "decode"
    return "unknown"


def normalize_model_name(name: str) -> str:
    """Remove workload suffix from model name."""
    return re.sub(r"-(prefill|decode)\b", "", name, flags=re.IGNORECASE)


def extract_tp_summary(tp_per_stage: str) -> str:
    """Extract TP summary from tp_per_stage string like '{PP_1:2, PP_2:2, ...}'."""
    if pd.isna(tp_per_stage) or not tp_per_stage:
        return "?"
    try:
        # Parse values from format like "{PP_1:2, PP_2:2, PP_3:1}"
        matches = re.findall(r":(\d+)", str(tp_per_stage))
        if matches:
            values = [int(m) for m in matches]
            unique_vals = sorted(set(values))
            if len(unique_vals) == 1:
                return str(unique_vals[0])
            return f"{min(values)}-{max(values)}"
    except Exception:
        pass
    return "?"


def extract_placement_label(placement: str) -> str:
    """Extract full placement info from placement string.

    Input format: '{PP_1:{g5.12xlarge#1:2}, PP_2:{g5.12xlarge#0:2}, PP_3:{g6e.4xlarge#0:1}}'
    Output: 'PP1:g5.12xl×2, PP2:g5.12xl×2, PP3:g6e.4xl×1'
    """
    if pd.isna(placement) or not placement:
        return ""
    try:
        # Parse format: PP_N:{device#id:count}
        # Match patterns like PP_1:{g5.12xlarge#1:2}
        pattern = r"PP_(\d+):\{([^#]+)[^:]*:(\d+)\}"
        matches = re.findall(pattern, str(placement))

        if not matches:
            return ""

        parts = []
        for pp_num, device, count in matches:
            # Shorten device name: g5.12xlarge -> g5.12xl
            device_short = re.sub(r"xlarge$", "xl", device.strip())
            parts.append(f"PP{pp_num}:{device_short}×{count}")

        return ", ".join(parts)
    except Exception:
        return ""


def format_throughput(value: float) -> str:
    """Format throughput values for display."""
    if value >= 1e6:
        return f"{value/1e6:.1f}M"
    if value >= 1e3:
        return f"{value/1e3:.1f}K"
    return f"{value:.0f}"


def format_cost(value: float) -> str:
    """Format cost values for display."""
    if value < 0.01:
        return f"${value:.4f}"
    if value < 1:
        return f"${value:.3f}"
    if value < 100:
        return f"${value:.2f}"
    return f"${value:.1f}"


def build_label(row: pd.Series) -> str:
    """Build a label showing model, batch, PP, TP, and full placement info."""
    model_name = normalize_model_name(str(row.get("model_name", "model")))
    bs = int(row.get("batch_size", 0))
    pp = int(row.get("pp", 0))
    tp_summary = extract_tp_summary(row.get("tp_per_stage", ""))
    num_gpus = int(row.get("num_gpus", 0))
    placement_label = extract_placement_label(row.get("placement", ""))

    # Shorten model name
    model_short = model_name.replace("llama3-", "L3-").replace("llama-", "L-")
    if len(model_short) > 10:
        model_short = model_short[:8] + ".."

    base_label = f"{model_short} | bs={bs} | PP={pp}, TP={tp_summary}, GPUs={num_gpus}"
    if placement_label:
        return f"{base_label} | {placement_label}"
    return base_label


def prepare_dataframe(df: pd.DataFrame, only_best: bool) -> pd.DataFrame:
    """Prepare and clean the dataframe for plotting."""
    df = df.copy()

    numeric_cols = [
        "total_tokens_per_sec",
        "dollar_per_million_token",
        "max_input_length",
        "max_output_length",
        "batch_size",
        "budget_tested",
        "pp",
        "num_gpus",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["workload"] = df.apply(infer_workload, axis=1)

    if only_best and "is_best" in df.columns:
        df = df[df["is_best"].astype(str).str.upper().eq("YES")]

    # Build label (contains PP, TP, GPUs, device info)
    df["label"] = df.apply(build_label, axis=1)

    # Round metrics for deduplication (avoid floating-point precision issues)
    df["_throughput_rounded"] = df["total_tokens_per_sec"].round(0)
    df["_cost_rounded"] = df["dollar_per_million_token"].round(2)

    # Deduplicate: keep one row per unique (workload, input/output, label, metrics)
    # This keeps different placements that have different performance
    dedup_cols = [
        "workload",
        "max_input_length",
        "max_output_length",
        "label",
        "_throughput_rounded",
        "_cost_rounded",
    ]
    existing_dedup_cols = [c for c in dedup_cols if c in df.columns]
    df = df.drop_duplicates(subset=existing_dedup_cols, keep="first")
    df = df.drop(columns=["_throughput_rounded", "_cost_rounded"], errors="ignore")

    return df


def plot_barh(
    ax,
    data: pd.DataFrame,
    metric: str,
    title: str,
    color: str,
) -> None:
    """Plot a horizontal bar chart with TP/PP annotations."""
    import numpy as np

    if data.empty:
        ax.set_visible(False)
        return

    # Sort: throughput descending (best first), cost ascending (best first)
    ascending = metric != "total_tokens_per_sec"
    data = data.sort_values(metric, ascending=ascending).reset_index(drop=True)

    n_bars = len(data)
    y_positions = np.arange(n_bars)
    bar_height = 0.65

    bars = ax.barh(
        y_positions,
        data[metric],
        height=bar_height,
        color=color,
        edgecolor=COLORS["bar_edge"],
        linewidth=0.5,
        alpha=0.85,
        zorder=3,
    )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(data["label"], fontsize=7, family="monospace")
    ax.set_title(title, fontweight="bold", pad=10, fontsize=11)

    if metric == "total_tokens_per_sec":
        ax.set_xlabel("Throughput (tokens/sec)")
    else:
        ax.set_xlabel("Cost ($/million tokens)")

    ax.invert_yaxis()
    ax.grid(True, axis="x", zorder=0)
    ax.grid(False, axis="y")

    # Add value annotations on bars
    max_val = data[metric].max()
    for i, bar in enumerate(bars):
        row = data.iloc[i]
        width = bar.get_width()

        # Format the metric value
        if metric == "total_tokens_per_sec":
            val_str = format_throughput(row[metric])
        else:
            val_str = format_cost(row[metric])

        # Position annotation inside or outside bar
        if width > max_val * 0.4:
            # Inside bar (right-aligned)
            ax.text(
                width - max_val * 0.01,
                bar.get_y() + bar_height / 2,
                val_str,
                va="center",
                ha="right",
                fontsize=8,
                color="white",
                fontweight="bold",
            )
        else:
            # Outside bar (left-aligned)
            ax.text(
                width + max_val * 0.01,
                bar.get_y() + bar_height / 2,
                val_str,
                va="center",
                ha="left",
                fontsize=8,
                color=COLORS["text"],
                fontweight="medium",
            )

    # Set x-axis to start at 0
    ax.set_xlim(left=0)


def plot_all(
    df: pd.DataFrame,
    output_path: Path,
    top_n: int,
    only_best: bool,
) -> Path:
    """Generate all plots and save to a single PDF."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError as exc:
        raise SystemExit(
            "Missing matplotlib. Install with: pip install matplotlib"
        ) from exc

    setup_matplotlib_style()

    workloads = [w for w in ["prefill", "decode"] if w in df["workload"].unique()]

    with PdfPages(output_path) as pdf:
        for workload in workloads:
            subset_w = df[df["workload"] == workload].copy()
            if subset_w.empty:
                continue

            io_pairs = (
                subset_w[["max_input_length", "max_output_length"]]
                .dropna()
                .drop_duplicates()
                .sort_values(["max_input_length", "max_output_length"])
            )

            for _, io_row in io_pairs.iterrows():
                in_len = int(io_row["max_input_length"])
                out_len = int(io_row["max_output_length"])
                subset = subset_w[
                    (subset_w["max_input_length"] == in_len)
                    & (subset_w["max_output_length"] == out_len)
                ].copy()

                if subset.empty:
                    continue

                top_throughput = subset.nlargest(top_n, "total_tokens_per_sec")
                top_cost = subset.nsmallest(top_n, "dollar_per_million_token")
                n_bars = max(len(top_throughput), len(top_cost), 1)

                # Calculate figure dimensions (wider to fit long placement labels)
                fig_height = max(4, 1.5 + n_bars * 0.5)
                fig_width = 18

                fig, axes = plt.subplots(
                    1, 2,
                    figsize=(fig_width, fig_height),
                    constrained_layout=True,
                )

                plot_barh(
                    axes[0],
                    top_throughput,
                    "total_tokens_per_sec",
                    f"Top {len(top_throughput)} by Throughput",
                    color=COLORS["throughput"],
                )
                plot_barh(
                    axes[1],
                    top_cost,
                    "dollar_per_million_token",
                    f"Top {len(top_cost)} by Cost Efficiency",
                    color=COLORS["cost"],
                )

                # Main title
                workload_label = workload.capitalize()
                title = f"{workload_label} Workload — Input={in_len}, Output={out_len}"
                if only_best:
                    title += " (Best Configurations Only)"

                fig.suptitle(title, fontsize=13, fontweight="bold")

                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate publication-quality plots from solution summaries.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/mnt/projects/LLM_placement_solver/solution_summaries_merged.csv"),
        help="Input CSV file with merged solution summaries",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top configurations to display per metric",
    )
    parser.add_argument(
        "--include-non-best",
        action="store_true",
        help="Include non-best configurations (default: only is_best=YES)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    df = pd.read_csv(args.input)
    df = prepare_dataframe(df, only_best=not args.include_non_best)

    required_cols = {"total_tokens_per_sec", "dollar_per_million_token"}
    missing = required_cols - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {', '.join(sorted(missing))}")

    output_path = args.input.resolve().parent / "solution_summaries_plot.pdf"

    output_path = plot_all(
        df,
        output_path=output_path,
        top_n=args.top_n,
        only_best=not args.include_non_best,
    )

    print(f"Generated: {output_path}")


if __name__ == "__main__":
    main()
