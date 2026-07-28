import gzip
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("results/mplconfig").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
MATRIX = DATA / "GSE54650_series_matrix.txt.gz"
ANNOT = DATA / "GPL6246.annot.gz"


def strip_quotes(value):
    return value.strip().strip('"')


def find_series_table(path):
    sample_titles = None
    begin = None
    end = None
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for i, line in enumerate(handle):
            if line.startswith("!Sample_title"):
                sample_titles = [strip_quotes(x) for x in line.rstrip("\n").split("\t")[1:]]
            elif line.startswith("!series_matrix_table_begin"):
                begin = i
            elif line.startswith("!series_matrix_table_end"):
                end = i
                break
    if sample_titles is None or begin is None or end is None:
        raise RuntimeError("Could not parse GEO series matrix structure.")
    return sample_titles, begin, end


def read_series_matrix(path):
    sample_titles, begin, end = find_series_table(path)
    nrows = end - begin - 2
    expr = pd.read_csv(
        path,
        sep="\t",
        compression="gzip",
        skiprows=begin + 1,
        nrows=nrows,
        dtype={"ID_REF": str},
    )
    id_col = expr.columns[0]
    expr = expr.rename(columns={id_col: "ID_REF"}).set_index("ID_REF")
    if len(sample_titles) != expr.shape[1]:
        raise RuntimeError("Sample title count does not match expression table columns.")
    expr.columns = sample_titles
    return expr.astype(float)


def read_platform_annotation(path):
    begin = None
    end = None
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for i, line in enumerate(handle):
            if line.startswith("!platform_table_begin"):
                begin = i
            elif line.startswith("!platform_table_end"):
                end = i
                break
    if begin is None:
        raise RuntimeError("Could not find platform table in annotation file.")
    nrows = None if end is None else end - begin - 2
    annot = pd.read_csv(
        path,
        sep="\t",
        compression="gzip",
        skiprows=begin + 1,
        nrows=nrows,
        dtype=str,
        low_memory=False,
    )
    annot = annot.rename(columns={"ID": "ID_REF", "Gene symbol": "gene_symbol"})
    annot["ID_REF"] = annot["ID_REF"].astype(str)
    annot["gene_symbol"] = annot["gene_symbol"].fillna("").astype(str)
    return annot[["ID_REF", "gene_symbol"]].drop_duplicates("ID_REF")


def first_harmonic_scores(log_expr, ct_hours):
    phase = 2.0 * np.pi * (ct_hours % 24.0) / 24.0
    design = np.column_stack([np.cos(phase), np.sin(phase)])
    y = log_expr - log_expr.mean(axis=0, keepdims=True)
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    fitted = np.dot(design, beta)
    ss_fit = np.sum(fitted**2, axis=0)
    ss_total = np.sum(y**2, axis=0)
    r2 = np.divide(ss_fit, ss_total, out=np.zeros_like(ss_fit), where=ss_total > 0)
    amplitude = np.sqrt(np.sum(beta**2, axis=0))
    peak_phase = np.mod(np.arctan2(beta[1], beta[0]), 2.0 * np.pi)
    peak_ct = np.mod(peak_phase * 24.0 / (2.0 * np.pi), 24.0)
    return r2, amplitude, peak_ct


def clr_transform(x):
    if np.any(x <= 0):
        positive = x[x > 0]
        pseudocount = positive.min() * 1e-3
        x = x + pseudocount
    log_x = np.log(x)
    return log_x - log_x.mean(axis=1, keepdims=True)


def acs_objective(cov, order):
    d = len(order)
    theta = 2.0 * np.pi * np.arange(d) / d
    c = np.sqrt(2.0 / d) * np.cos(theta)
    s = np.sqrt(2.0 / d) * np.sin(theta)
    ordered = cov[np.ix_(order, order)]
    return float(np.dot(c, np.dot(ordered, c)) + np.dot(s, np.dot(ordered, s)))


def spectral_acs_order(cov, rng, n_swaps=4000):
    evals, evecs = np.linalg.eigh(cov)
    xy = evecs[:, -2:] * np.sqrt(np.maximum(evals[-2:], 0.0))
    order = np.argsort(np.arctan2(xy[:, 1], xy[:, 0]))

    best = acs_objective(cov, order)
    order = order.copy()
    d = len(order)
    for _ in range(n_swaps):
        i, j = rng.choice(d, size=2, replace=False)
        candidate = order.copy()
        candidate[i], candidate[j] = candidate[j], candidate[i]
        score = acs_objective(cov, candidate)
        if score > best:
            order = candidate
            best = score
    return order, best


def align_angles(u, v, ct_hours):
    z = u + 1j * v
    true_phase = 2.0 * np.pi * (ct_hours % 24.0) / 24.0
    best = None
    for reflected in [False, True]:
        zz = np.conj(z) if reflected else z
        raw = np.angle(zz)
        delta = np.angle(np.mean(np.exp(1j * (raw - true_phase))))
        aligned = zz * np.exp(-1j * delta)
        aligned_angle = np.angle(aligned)
        circular_residual = np.angle(np.exp(1j * (aligned_angle - true_phase)))
        r = np.abs(np.mean(np.exp(1j * circular_residual)))
        mean_abs_error_hours = np.mean(np.abs(circular_residual)) * 24.0 / (2.0 * np.pi)
        candidate = (r, -mean_abs_error_hours, aligned, reflected, delta, circular_residual)
        if best is None or candidate > best:
            best = candidate
    r, neg_err, aligned, reflected, delta, residual = best
    return aligned.real, aligned.imag, r, -neg_err, reflected, delta, residual


def plot_results(sample_df, feature_df, out_png, out_svg):
    fig = plt.figure(figsize=(12.5, 5.8), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0])
    cmap = plt.get_cmap("twilight_shifted")

    ax = fig.add_subplot(gs[0, 0])
    sc = ax.scatter(
        sample_df["acs_u_aligned"],
        sample_df["acs_v_aligned"],
        c=sample_df["ct_mod24"],
        cmap=cmap,
        vmin=0,
        vmax=24,
        s=86,
        edgecolor="black",
        linewidth=0.8,
        zorder=3,
    )
    sample_radius = float(np.median(sample_df["acs_radius"]))
    sample_circle = plt.Circle(
        (0, 0),
        sample_radius,
        fill=False,
        color="0.45",
        linestyle="--",
        linewidth=1.1,
        alpha=0.75,
        zorder=0,
    )
    ax.add_patch(sample_circle)
    chronological = sample_df.sort_values("ct")
    ax.plot(
        chronological["acs_u_aligned"],
        chronological["acs_v_aligned"],
        color="0.65",
        linewidth=1.1,
        zorder=1,
    )
    for _, row in sample_df.iterrows():
        ax.text(
            row["acs_u_aligned"],
            row["acs_v_aligned"],
            f"{int(row['ct'])}",
            fontsize=7,
            ha="center",
            va="center",
            color="white",
            zorder=4,
        )
    ax.axhline(0, color="0.85", linewidth=0.8)
    ax.axvline(0, color="0.85", linewidth=0.8)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title("Liver samples projected by clrcycle")
    ax.set_xlabel("ACS cosine coordinate")
    ax.set_ylabel("ACS sine coordinate")

    ax2 = fig.add_subplot(gs[0, 1])
    theta = 2.0 * np.pi * feature_df["acs_position"] / len(feature_df)
    radius = 1.0
    fx = radius * np.cos(theta)
    fy = radius * np.sin(theta)
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
    label_genes = {
        "Arntl",
        "Clock",
        "Cry1",
        "Cry2",
        "Per1",
        "Per2",
        "Per3",
        "Nr1d1",
        "Dbp",
        "Rorc",
        "Nampt",
    }
    label_df = feature_df[feature_df["gene_symbol"].isin(label_genes)].copy()
    label_df = label_df[label_df["gene_symbol"].str.len() > 0]
    for _, row in label_df.iterrows():
        angle = 2.0 * np.pi * row["acs_position"] / len(feature_df)
        x = 1.12 * np.cos(angle)
        y = 1.12 * np.sin(angle)
        ha = "left" if x >= 0 else "right"
        ax2.text(x, y, row["gene_symbol"], fontsize=6.4, ha=ha, va="center")
    circle = plt.Circle((0, 0), 1.0, fill=False, color="0.75", linewidth=0.9)
    ax2.add_patch(circle)
    ax2.set_aspect("equal")
    ax2.set_title("Learned feature circle, colored by peak CT")
    ax2.set_xticks([])
    ax2.set_yticks([])
    for spine in ax2.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(sc, ax=[ax, ax2], orientation="horizontal", fraction=0.08, pad=0.08)
    cbar.set_label("Circadian time modulo 24 h")
    cbar.set_ticks([0, 6, 12, 18, 24])
    fig.suptitle("clrcycle demo on GSE54650 mouse liver circadian atlas", fontsize=14)
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_svg)
    plt.close(fig)


def main():
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "mplconfig").mkdir(exist_ok=True)

    expr = read_series_matrix(MATRIX)
    annot = read_platform_annotation(ANNOT)

    liver_cols = [col for col in expr.columns if col.startswith("Liv_CT")]
    liver_cols = sorted(liver_cols, key=lambda name: int(re.search(r"CT(\d+)", name).group(1)))
    liver = expr[liver_cols].T
    ct = np.array([int(re.search(r"CT(\d+)", name).group(1)) for name in liver.index], dtype=float)

    valid = np.isfinite(liver.to_numpy()).all(axis=0) & (liver.to_numpy().min(axis=0) > 0)
    liver = liver.loc[:, valid]

    log_expr = np.log(liver.to_numpy())
    r2, amplitude, peak_ct = first_harmonic_scores(log_expr, ct)
    scores = pd.DataFrame(
        {
            "ID_REF": liver.columns.astype(str),
            "harmonic_r2": r2,
            "harmonic_amplitude": amplitude,
            "peak_ct": peak_ct,
            "mean_expression": liver.mean(axis=0).to_numpy(),
        }
    ).merge(annot, on="ID_REF", how="left")
    scores["gene_symbol"] = scores["gene_symbol"].fillna("")
    scores = scores[scores["gene_symbol"].str.len() > 0].copy()
    scores = scores.sort_values(["harmonic_r2", "harmonic_amplitude"], ascending=False)

    top_n = 240
    selected = scores.head(top_n).copy()
    selected_ids = selected["ID_REF"].tolist()

    z = clr_transform(liver[selected_ids].to_numpy())
    z_centered = z - z.mean(axis=0, keepdims=True)
    cov = np.dot(z_centered.T, z_centered) / (z_centered.shape[0] - 1)
    if not np.isfinite(cov).all():
        raise RuntimeError("Non-finite values appeared in the clr covariance matrix.")

    rng = np.random.default_rng(17)
    order, objective = spectral_acs_order(cov, rng)
    ordered_ids = [selected_ids[i] for i in order]

    d = len(order)
    theta = 2.0 * np.pi * np.arange(d) / d
    c = np.sqrt(2.0 / d) * np.cos(theta)
    s = np.sqrt(2.0 / d) * np.sin(theta)
    u = np.dot(z_centered[:, order], c)
    v = np.dot(z_centered[:, order], s)
    if not np.isfinite(u).all() or not np.isfinite(v).all():
        raise RuntimeError("Non-finite values appeared in the ACS projection.")

    u_aligned, v_aligned, phase_r, mean_abs_error, reflected, delta, residual = align_angles(u, v, ct)

    evals = np.linalg.eigvalsh(cov)
    lambda1, lambda2 = evals[-1], evals[-2]
    rho = objective / (lambda1 + lambda2)

    sample_df = pd.DataFrame(
        {
            "sample": liver.index,
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
    feature_df["acs_position"] = np.arange(d)
    feature_df["acs_angle"] = theta
    feature_df = feature_df[
        [
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

    sample_df.to_csv(RESULTS / "liver_clr_acs_sample_coordinates.csv", index=False)
    feature_df.to_csv(RESULTS / "liver_clr_acs_feature_order.csv", index=False)
    scores.to_csv(RESULTS / "liver_harmonic_feature_scores.csv", index=False)
    plot_results(
        sample_df,
        feature_df,
        RESULTS / "liver_clr_acs_projection.png",
        RESULTS / "liver_clr_acs_projection.svg",
    )

    print("Dataset: GSE54650 liver subset")
    print(f"Samples: {liver.shape[0]}")
    print(f"Candidate positive probes: {liver.shape[1]}")
    print(f"Selected rhythmic mapped probes: {d}")
    print(f"ACS objective: {objective:.6f}")
    print(f"rho_ACS heuristic: {rho:.6f}")
    print(f"Circular phase agreement R: {phase_r:.6f}")
    print(f"Mean absolute phase error: {mean_abs_error:.3f} hours")
    print(f"Alignment reflected: {reflected}")
    print("Wrote results/liver_clr_acs_projection.png")


if __name__ == "__main__":
    main()
