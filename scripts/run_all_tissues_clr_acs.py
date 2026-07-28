import argparse
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("results/mplconfig").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_liver_clr_acs import (  # noqa: E402
    ANNOT,
    MATRIX,
    RESULTS,
    acs_objective,
    align_angles,
    clr_transform,
    first_harmonic_scores,
    read_platform_annotation,
    read_series_matrix,
    spectral_acs_order,
)


TISSUE_LABELS = {
    "Adr": "Adrenal",
    "Aor": "Aorta",
    "Bstm": "Brainstem",
    "BFat": "Brown fat",
    "Cer": "Cerebellum",
    "Hrt": "Heart",
    "Hyp": "Hypothalamus",
    "Kid": "Kidney",
    "Liv": "Liver",
    "Lun": "Lung",
    "Mus": "Muscle",
    "WFat": "White fat",
}


def tissue_prefixes(columns):
    prefixes = []
    seen = set()
    for col in columns:
        match = re.match(r"([A-Za-z]+)_CT\d+$", col)
        if match and match.group(1) not in seen:
            prefixes.append(match.group(1))
            seen.add(match.group(1))
    return prefixes


def pca_phase_from_features(tissue, feature_ids):
    z = clr_transform(tissue[feature_ids].to_numpy())
    z = z - z.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(z, full_matrices=False)
    coords = u[:, :2] * s[:2]
    return np.angle(coords[:, 0] + 1j * coords[:, 1])


def harmonic_scores_against_phase(log_expr, phase):
    design = np.column_stack([np.cos(phase), np.sin(phase)])
    y = log_expr - log_expr.mean(axis=0, keepdims=True)
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    fitted = np.dot(design, beta)
    ss_fit = np.sum(fitted**2, axis=0)
    ss_total = np.sum(y**2, axis=0)
    r2 = np.divide(ss_fit, ss_total, out=np.zeros_like(ss_fit), where=ss_total > 0)
    amplitude = np.sqrt(np.sum(beta**2, axis=0))
    peak_phase = np.mod(np.arctan2(beta[1], beta[0]), 2.0 * np.pi)
    return r2, amplitude, peak_phase


def select_features(tissue, annot, ct, top_n, mode):
    if mode == "rhythmic":
        log_expr = np.log(tissue.to_numpy())
        r2, amplitude, peak_ct = first_harmonic_scores(log_expr, ct)
        scores = pd.DataFrame(
            {
                "ID_REF": tissue.columns.astype(str),
                "harmonic_r2": r2,
                "harmonic_amplitude": amplitude,
                "peak_ct": peak_ct,
                "mean_expression": tissue.mean(axis=0).to_numpy(),
            }
        ).merge(annot, on="ID_REF", how="left")
        scores["selection_score"] = scores["harmonic_r2"]
        scores = scores.sort_values(["harmonic_r2", "harmonic_amplitude"], ascending=False)
    elif mode == "variance":
        values = tissue.to_numpy()
        log_expr = np.log(values)
        log_var = log_expr.var(axis=0, ddof=1)
        log_mean = log_expr.mean(axis=0)
        # Peak CT is computed only for annotation/plot coloring, not for feature selection.
        r2, amplitude, peak_ct = first_harmonic_scores(log_expr, ct)
        scores = pd.DataFrame(
            {
                "ID_REF": tissue.columns.astype(str),
                "selection_score": log_var,
                "log_variance": log_var,
                "log_mean": log_mean,
                "harmonic_r2": r2,
                "harmonic_amplitude": amplitude,
                "peak_ct": peak_ct,
                "mean_expression": tissue.mean(axis=0).to_numpy(),
            }
        ).merge(annot, on="ID_REF", how="left")
        scores = scores.sort_values(["selection_score", "mean_expression"], ascending=False)
    elif mode == "self_periodic":
        values = tissue.to_numpy()
        log_expr = np.log(values)
        log_var = log_expr.var(axis=0, ddof=1)
        base_scores = pd.DataFrame(
            {
                "ID_REF": tissue.columns.astype(str),
                "log_variance": log_var,
                "mean_expression": tissue.mean(axis=0).to_numpy(),
            }
        ).merge(annot, on="ID_REF", how="left")
        base_scores["gene_symbol"] = base_scores["gene_symbol"].fillna("")
        base_scores = base_scores[base_scores["gene_symbol"].str.len() > 0].copy()

        pool_n = min(3000, len(base_scores))
        phase = pca_phase_from_features(
            tissue,
            base_scores.sort_values(["log_variance", "mean_expression"], ascending=False)
            .head(pool_n)["ID_REF"]
            .tolist(),
        )

        indexed_expr = tissue[base_scores["ID_REF"].tolist()]
        for _ in range(4):
            self_r2, self_amp, self_peak_phase = harmonic_scores_against_phase(
                np.log(indexed_expr.to_numpy()),
                phase,
            )
            iteration_scores = base_scores.copy()
            iteration_scores["self_periodic_r2"] = self_r2
            iteration_scores["self_periodic_amplitude"] = self_amp
            iteration_scores["self_peak_phase"] = self_peak_phase
            iteration_scores["selection_score"] = iteration_scores["self_periodic_r2"]
            selected_ids = (
                iteration_scores.sort_values(["selection_score", "self_periodic_amplitude"], ascending=False)
                .head(top_n)["ID_REF"]
                .tolist()
            )
            phase = pca_phase_from_features(tissue, selected_ids)

        r2, amplitude, peak_ct = first_harmonic_scores(log_expr, ct)
        true_time_scores = pd.DataFrame(
            {
                "ID_REF": tissue.columns.astype(str),
                "harmonic_r2": r2,
                "harmonic_amplitude": amplitude,
                "peak_ct": peak_ct,
                "mean_expression": tissue.mean(axis=0).to_numpy(),
            }
        )
        scores = iteration_scores.merge(true_time_scores, on=["ID_REF", "mean_expression"], how="left")
        scores = scores.sort_values(["selection_score", "self_periodic_amplitude"], ascending=False)
    elif mode == "repeat_periodic":
        values = tissue.to_numpy()
        log_expr = np.log(values)
        n_samples = log_expr.shape[0]
        if n_samples % 2 != 0:
            raise ValueError("repeat_periodic selection expects two equal cycles.")
        half = n_samples // 2
        first = log_expr[:half, :]
        second = log_expr[half:, :]
        repeated_mean = 0.5 * (first + second)
        across_phase_var = repeated_mean.var(axis=0, ddof=1)
        within_phase_var = 0.5 * ((first - repeated_mean) ** 2 + (second - repeated_mean) ** 2).mean(axis=0)
        repeat_score = across_phase_var / (within_phase_var + 1e-8)
        cycle_corr = np.array(
            [
                np.corrcoef(first[:, j], second[:, j])[0, 1]
                if np.std(first[:, j]) > 0 and np.std(second[:, j]) > 0
                else 0.0
                for j in range(log_expr.shape[1])
            ]
        )
        cycle_corr = np.nan_to_num(cycle_corr, nan=0.0)
        # CT-based quantities are only for annotation and downstream validation.
        r2, amplitude, peak_ct = first_harmonic_scores(log_expr, ct)
        scores = pd.DataFrame(
            {
                "ID_REF": tissue.columns.astype(str),
                "selection_score": repeat_score,
                "repeat_score": repeat_score,
                "cycle_corr": cycle_corr,
                "across_phase_var": across_phase_var,
                "within_phase_var": within_phase_var,
                "harmonic_r2": r2,
                "harmonic_amplitude": amplitude,
                "peak_ct": peak_ct,
                "mean_expression": tissue.mean(axis=0).to_numpy(),
            }
        ).merge(annot, on="ID_REF", how="left")
        scores = scores.sort_values(["selection_score", "cycle_corr"], ascending=False)
    else:
        raise ValueError(f"Unknown feature selection mode: {mode}")

    scores["gene_symbol"] = scores["gene_symbol"].fillna("")
    scores = scores[scores["gene_symbol"].str.len() > 0].copy()
    selected = scores.head(top_n).copy()
    return selected, scores


def analyze_tissue(expr, annot, prefix, top_n=240, seed=17, selection_mode="rhythmic"):
    tissue_cols = [col for col in expr.columns if col.startswith(f"{prefix}_CT")]
    tissue_cols = sorted(
        tissue_cols,
        key=lambda name: int(re.search(r"CT(\d+)", name).group(1)),
    )
    tissue = expr[tissue_cols].T
    ct = np.array([int(re.search(r"CT(\d+)", name).group(1)) for name in tissue.index], dtype=float)

    values = tissue.to_numpy()
    valid = np.isfinite(values).all(axis=0) & (values.min(axis=0) > 0)
    tissue = tissue.loc[:, valid]

    selected, scores = select_features(tissue, annot, ct, top_n, selection_mode)
    selected_ids = selected["ID_REF"].tolist()
    z = clr_transform(tissue[selected_ids].to_numpy())
    z_centered = z - z.mean(axis=0, keepdims=True)
    cov = np.dot(z_centered.T, z_centered) / (z_centered.shape[0] - 1)
    if not np.isfinite(cov).all():
        raise RuntimeError(f"Non-finite covariance for {prefix}.")

    rng = np.random.default_rng(seed)
    order, objective = spectral_acs_order(cov, rng)
    ordered_ids = [selected_ids[i] for i in order]

    d = len(order)
    theta = 2.0 * np.pi * np.arange(d) / d
    c = np.sqrt(2.0 / d) * np.cos(theta)
    s = np.sqrt(2.0 / d) * np.sin(theta)
    u = np.dot(z_centered[:, order], c)
    v = np.dot(z_centered[:, order], s)
    u_aligned, v_aligned, phase_r, mean_abs_error, reflected, delta, residual = align_angles(u, v, ct)

    evals = np.linalg.eigvalsh(cov)
    rho = objective / (evals[-1] + evals[-2])

    sample_df = pd.DataFrame(
        {
            "tissue": prefix,
            "sample": tissue.index,
            "ct": ct,
            "ct_mod24": ct % 24.0,
            "acs_u": u,
            "acs_v": v,
            "acs_u_aligned": u_aligned,
            "acs_v_aligned": v_aligned,
            "acs_angle_aligned": np.mod(np.angle(u_aligned + 1j * v_aligned), 2.0 * np.pi),
            "acs_radius": np.sqrt(u_aligned**2 + v_aligned**2),
            "phase_residual_hours": residual * 24.0 / (2.0 * np.pi),
        }
    )

    feature_df = selected.set_index("ID_REF").loc[ordered_ids].reset_index()
    feature_df["tissue"] = prefix
    feature_df["acs_position"] = np.arange(d)
    feature_df["acs_angle"] = theta
    feature_df = feature_df[
        [
            "tissue",
            "ID_REF",
            "gene_symbol",
            "acs_position",
            "acs_angle",
            "harmonic_r2",
            "harmonic_amplitude",
            "peak_ct",
            "mean_expression",
        ]
    ]

    peak_row = sample_df.loc[sample_df["acs_radius"].idxmax()]
    trough_row = sample_df.loc[sample_df["acs_radius"].idxmin()]
    metrics = {
        "tissue": prefix,
        "label": TISSUE_LABELS.get(prefix, prefix),
        "n_samples": tissue.shape[0],
        "n_candidate_positive_probes": tissue.shape[1],
        "n_selected_probes": d,
        "selection_mode": selection_mode,
        "acs_objective": objective,
        "rho_acs": rho,
        "phase_agreement_r": phase_r,
        "mean_abs_phase_error_hours": mean_abs_error,
        "alignment_reflected": reflected,
        "peak_radius_ct": peak_row["ct"],
        "peak_radius_ct_mod24": peak_row["ct_mod24"],
        "peak_radius": peak_row["acs_radius"],
        "trough_radius_ct": trough_row["ct"],
        "trough_radius_ct_mod24": trough_row["ct_mod24"],
        "trough_radius": trough_row["acs_radius"],
    }
    return sample_df, feature_df, scores, metrics


def draw_sample_panel(ax, sample_df, title, cmap, label_points=True):
    sc = ax.scatter(
        sample_df["acs_u_aligned"],
        sample_df["acs_v_aligned"],
        c=sample_df["ct_mod24"],
        cmap=cmap,
        vmin=0,
        vmax=24,
        s=64 if label_points else 26,
        edgecolor="black" if label_points else "none",
        linewidth=0.7,
        zorder=3,
    )
    sample_radius = float(np.median(sample_df["acs_radius"]))
    ax.add_patch(
        plt.Circle(
            (0, 0),
            sample_radius,
            fill=False,
            color="0.5",
            linestyle="--",
            linewidth=0.9,
            alpha=0.7,
            zorder=0,
        )
    )
    chronological = sample_df.sort_values("ct")
    ax.plot(
        chronological["acs_u_aligned"],
        chronological["acs_v_aligned"],
        color="0.65",
        linewidth=1.0,
        zorder=1,
    )
    if label_points:
        for _, row in sample_df.iterrows():
            ax.text(
                row["acs_u_aligned"],
                row["acs_v_aligned"],
                f"{int(row['ct'])}",
                fontsize=6.5,
                ha="center",
                va="center",
                color="white",
                zorder=4,
            )
    ax.axhline(0, color="0.88", linewidth=0.8)
    ax.axvline(0, color="0.88", linewidth=0.8)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(title)
    return sc


def plot_tissue(sample_df, feature_df, metrics, out_png, out_svg):
    label = metrics["label"]
    cmap = plt.get_cmap("twilight_shifted")
    fig = plt.figure(figsize=(12.5, 5.8), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0])

    ax = fig.add_subplot(gs[0, 0])
    sc = draw_sample_panel(ax, sample_df, f"{label} samples projected by clrcycle", cmap)
    ax.set_xlabel("ACS cosine coordinate")
    ax.set_ylabel("ACS sine coordinate")

    ax2 = fig.add_subplot(gs[0, 1])
    theta = 2.0 * np.pi * feature_df["acs_position"] / len(feature_df)
    fx = np.cos(theta)
    fy = np.sin(theta)
    ax2.scatter(
        fx,
        fy,
        c=feature_df["peak_ct"],
        cmap=cmap,
        vmin=0,
        vmax=24,
        s=16,
        alpha=0.88,
        linewidth=0,
    )
    label_genes = {"Arntl", "Clock", "Cry1", "Cry2", "Per1", "Per2", "Per3", "Nr1d1", "Dbp", "Rorc", "Nampt"}
    for _, row in feature_df[feature_df["gene_symbol"].isin(label_genes)].iterrows():
        angle = 2.0 * np.pi * row["acs_position"] / len(feature_df)
        x = 1.12 * np.cos(angle)
        y = 1.12 * np.sin(angle)
        ax2.text(x, y, row["gene_symbol"], fontsize=6.4, ha="left" if x >= 0 else "right", va="center")
    ax2.add_patch(plt.Circle((0, 0), 1.0, fill=False, color="0.75", linewidth=0.9))
    ax2.set_aspect("equal")
    ax2.set_title("Learned feature circle, colored by peak CT")
    ax2.set_xticks([])
    ax2.set_yticks([])
    for spine in ax2.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(sc, ax=[ax, ax2], orientation="horizontal", fraction=0.08, pad=0.08)
    cbar.set_label("Circadian time modulo 24 h")
    cbar.set_ticks([0, 6, 12, 18, 24])
    fig.suptitle(
        f"GSE54650 {label} ({metrics['selection_mode']} selection): R={metrics['phase_agreement_r']:.3f}, "
        f"mean phase error={metrics['mean_abs_phase_error_hours']:.2f} h",
        fontsize=14,
    )
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_svg)
    plt.close(fig)


def plot_summary_grid(sample_tables, metrics_df, out_png, out_svg):
    cmap = plt.get_cmap("twilight_shifted")
    fig, axes = plt.subplots(3, 4, figsize=(14, 9), constrained_layout=True)
    sc = None
    for ax, (prefix, sample_df) in zip(axes.ravel(), sample_tables.items()):
        row = metrics_df[metrics_df["tissue"] == prefix].iloc[0]
        title = (
            f"{row['label']}\n"
            f"R={row['phase_agreement_r']:.2f}, peak CT{int(row['peak_radius_ct_mod24'])}"
        )
        sc = draw_sample_panel(ax, sample_df, title, cmap, label_points=False)
        ax.set_xticks([])
        ax.set_yticks([])
    cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), orientation="horizontal", fraction=0.05, pad=0.04)
    cbar.set_label("Circadian time modulo 24 h")
    cbar.set_ticks([0, 6, 12, 18, 24])
    mode = metrics_df["selection_mode"].iloc[0]
    fig.suptitle(f"clrcycle sample projections across GSE54650 tissues ({mode} feature selection)", fontsize=16)
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_svg)
    plt.close(fig)


def plot_big_panel_grid(sample_tables, metrics_df, out_png, out_pdf):
    cmap = plt.get_cmap("twilight_shifted")
    fig, axes = plt.subplots(3, 4, figsize=(20, 14), constrained_layout=True)
    panel_letters = list("ABCDEFGHIJKL")
    sc = None
    for letter, ax, (prefix, sample_df) in zip(panel_letters, axes.ravel(), sample_tables.items()):
        row = metrics_df[metrics_df["tissue"] == prefix].iloc[0]
        sc = draw_sample_panel(ax, sample_df, row["label"], cmap, label_points=True)
        ax.text(
            0.02,
            0.96,
            letter,
            transform=ax.transAxes,
            fontsize=16,
            fontweight="bold",
            ha="left",
            va="top",
        )
        ax.text(
            0.98,
            0.04,
            f"R={row['phase_agreement_r']:.2f}\npeak CT{int(row['peak_radius_ct_mod24'])}",
            transform=ax.transAxes,
            fontsize=9,
            ha="right",
            va="bottom",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 2},
        )
        ax.set_xlabel("ACS cosine")
        ax.set_ylabel("ACS sine")
    cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), orientation="horizontal", fraction=0.04, pad=0.035)
    cbar.set_label("Circadian time modulo 24 h")
    cbar.set_ticks([0, 6, 12, 18, 24])
    mode = metrics_df["selection_mode"].iloc[0]
    fig.suptitle(f"clrcycle sample projections across GSE54650 tissues ({mode} feature selection)", fontsize=22)
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_pdf)
    plt.close(fig)


def run_all(selection_mode, out_name, top_n=240, results_dir=RESULTS):
    out_dir = results_dir / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (RESULTS / "mplconfig").mkdir(exist_ok=True)

    expr = read_series_matrix(MATRIX)
    annot = read_platform_annotation(ANNOT)
    prefixes = tissue_prefixes(expr.columns)

    sample_tables = {}
    feature_tables = []
    metric_rows = []
    for idx, prefix in enumerate(prefixes):
        sample_df, feature_df, scores, metrics = analyze_tissue(
            expr,
            annot,
            prefix,
            top_n=top_n,
            seed=17 + idx,
            selection_mode=selection_mode,
        )
        sample_tables[prefix] = sample_df
        feature_tables.append(feature_df)
        metric_rows.append(metrics)

        stem = f"{prefix.lower()}_clr_acs_projection"
        sample_df.to_csv(out_dir / f"{prefix.lower()}_sample_coordinates.csv", index=False)
        feature_df.to_csv(out_dir / f"{prefix.lower()}_feature_order.csv", index=False)
        scores.to_csv(out_dir / f"{prefix.lower()}_harmonic_feature_scores.csv", index=False)
        plot_tissue(sample_df, feature_df, metrics, out_dir / f"{stem}.png", out_dir / f"{stem}.svg")
        print(
            f"{prefix:>4} {metrics['label']:<12} "
            f"{selection_mode:<8} "
            f"R={metrics['phase_agreement_r']:.3f} "
            f"err={metrics['mean_abs_phase_error_hours']:.2f}h "
            f"peak=CT{int(metrics['peak_radius_ct_mod24'])}"
        )

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(out_dir / "all_tissue_metrics.csv", index=False)
    pd.concat(feature_tables, ignore_index=True).to_csv(out_dir / "all_tissue_feature_orders.csv", index=False)
    pd.concat(sample_tables.values(), ignore_index=True).to_csv(out_dir / "all_tissue_sample_coordinates.csv", index=False)
    plot_summary_grid(
        sample_tables,
        metrics_df,
        out_dir / "all_tissues_sample_projection_grid.png",
        out_dir / "all_tissues_sample_projection_grid.svg",
    )
    plot_big_panel_grid(
        sample_tables,
        metrics_df,
        out_dir / "all_tissues_sample_projection_panels.png",
        out_dir / "all_tissues_sample_projection_panels.pdf",
    )

    print(f"Wrote {len(prefixes)} tissue plots to {out_dir}")
    print(f"Wrote summary grid to {out_dir / 'all_tissues_sample_projection_grid.png'}")
    print(f"Wrote big panel figure to {out_dir / 'all_tissues_sample_projection_panels.png'}")


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce the clrcycle circadian analyses in GSE54650."
    )
    parser.add_argument(
        "--analysis",
        choices=("both", "supervised", "label-free"),
        default="both",
        help="Analysis to run (default: both analyses used in the manuscript).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS,
        help="Directory in which to write generated outputs (default: results/).",
    )
    args = parser.parse_args()
    results_dir = args.results_dir.resolve()

    if args.analysis in ("both", "supervised"):
        run_all("rhythmic", "all_tissues", top_n=240, results_dir=results_dir)
    if args.analysis in ("both", "label-free"):
        run_all(
            "repeat_periodic",
            "all_tissues_unsupervised_repeat_periodic_96",
            top_n=96,
            results_dir=results_dir,
        )


if __name__ == "__main__":
    main()
