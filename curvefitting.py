import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sympy import gamma

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

import warnings
warnings.filterwarnings('ignore')


# ── constants ─────────────────────────────────────────────
GAMMA = 0.5   # guessing rate
TARGET = 0.75 # JND definition

# ── weibull function (NO lapse) ───────────────────────────
def weibull(x, alpha, beta, gamma=GAMMA):
    return gamma + (1 - gamma) * (1 - np.exp(-(x / alpha) ** beta)) # weibull
    # return gamma + (1 - gamma) * (1 - np.exp(-np.exp((x - alpha) / beta))) # gumbell (log-weibull)
    # return gamma + (1 - gamma - lapse) / (1 + np.exp(-(x - alpha) / beta)) # logistic (for comparison)


# ── negative log likelihood ───────────────────────────────
def neg_ll(params, x, y):
    log_alpha, log_beta = params

    alpha = np.exp(log_alpha)
    beta  = np.exp(log_beta)

    p = weibull(x, alpha, beta)
    p = np.clip(p, 0.5, 1 - 1e-9)

    return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))


# ── compute JND from fitted params ────────────────────────
def compute_jnd(alpha, beta, target=TARGET, gamma=GAMMA):
    """
    Solve for x where weibull(x) = target
    """
    val = -np.log(1 - (target - gamma) / (1 - gamma))
    return alpha * (val ** (1 / beta)) # weibull
    # return beta * np.log(-np.log(0.25/(1-gamma))) + alpha # gumbell


# ── main fitting function ─────────────────────────────────
def fit_all_participants(csv_path, output_csv):

    df = pd.read_csv(csv_path)

    # ensure binary
    df["gotItRight"] = df["gotItRight"].astype(int)

    # compute delta
    df["delta"] = np.abs(df["rbase"] - df["rv"])

    results_rows = []

    # group by participant + condition
    grouped = df.groupby(["participant", "rbase", "approach"])

    for (participant, rbase, approach), subset in grouped:

        if len(subset) < 3:
            continue  # skip too small

        x = subset["delta"].values
        y = subset["gotItRight"].values

        # initial guess
        p0 = [np.log(0.2), np.log(2.0)]

        try:
            res = minimize(
                neg_ll,
                p0,
                args=(x, y),
                method="L-BFGS-B",
                bounds=[(np.log(1e-4), np.log(1.0)),
                        (np.log(0.5), np.log(20))]
            )

            if not res.success:
                continue

            log_alpha, log_beta = res.x
            alpha = np.exp(log_alpha)
            beta  = np.exp(log_beta)

            # compute JND
            jnd_calc = compute_jnd(alpha, beta)

            if jnd_calc > 1:
                print(f"Bad fit: Participant={participant}, r={r}, jnd={jnd_calc:.3f}")
                continue

        except Exception:
            continue

        # ── add row per original trial ─────────────────────
        for _, row in subset.iterrows():
            results_rows.append({
                "participant": participant,
                "rbase": rbase,
                "rv": row["rv"],
                "approach": approach,
                "correctChoice": row["correctChoice"],
                "currentChoice": row["currentChoice"],
                "gotItRight": row["gotItRight"],
                "jndFromHarrison": row["jnd"],
                "calculatedJND": jnd_calc,
                "alpha": alpha,
                "beta": beta,
                "delta": row["delta"],
                "jnd_diff": row["jnd"] - jnd_calc
            })

    # save CSV
    df_out = pd.DataFrame(results_rows)
    df_out.to_csv(output_csv, index=False)

    df_diff = (
        df_out
        .groupby(["participant", "rbase", "approach"])["jnd_diff"]
        .mean()
        .reset_index()
    )

    df_diff["participant_idx"] = range(1, len(df_diff) + 1)

    print(f"Saved results to {output_csv}")
    return df_out, df_diff


def fit_model_by_rbase(input_csv, output_csv):

    df = pd.read_csv(input_csv)

    # rbase_values = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # For positive, use [0.3, 0.4, 0.5, 0.6, 0.7, 0.8] change here!!!!
    rbase_values = [-0.3, -0.4, -0.5, -0.6, -0.7, -0.8] # For negative, use [-0.3, -0.4, -0.5, -0.6, -0.7, -0.8]

    results = []

    print("\nModel fitting: ")

    for r in rbase_values:

        subset = df[
            (np.isclose(df["corr1 (Left)"], r, atol = 0.00)) |
            (np.isclose(df["corr2 (Right)"], r, atol = 0.00))
        ].copy()

        if len(subset) < 3:
            print(f"Skipping r={r}, too few samples")
            continue

        x = subset["delta_corr"].values
        y = subset["accuracy"].values

        # initial guess
        p0 = [np.log(np.median(x)), np.log(2.0)]

        try:
            res = minimize(
                neg_ll,
                p0,
                args=(x, y),
                method="L-BFGS-B",
                bounds=[(np.log(1e-4), np.log(1.0)),
                        (np.log(0.5), np.log(20))]
            )

            if not res.success:
                print(f"Fit failed for r={r}")
                continue

            log_alpha, log_beta = res.x
            alpha = np.exp(log_alpha)
            beta  = np.exp(log_beta)

            jnd = compute_jnd(alpha, beta)

        except Exception:
            print(f"Error at r={r}")
            continue

        print(f"r={r}: alpha={alpha:.4f}, beta={beta:.3f}, JND={jnd:.4f}")

        # save row-wise (like your Harrison output)
        for _, row in subset.iterrows():
            results.append({
                "corr1": row["corr1 (Left)"],
                "corr2": row["corr2 (Right)"],
                "correct": row["correct"],
                "modelResponse": row["modelResponse"],
                "accuracy": row["accuracy"],
                "delta_corr": row["delta_corr"],
                "rbase": r,
                "alpha": alpha,
                "beta": beta,
                "jnd": jnd
            })

    df_out = pd.DataFrame(results)
    df_out.to_csv(output_csv, index=False)

    print(f"Saved model fits → {output_csv}")
    return df_out


def fit_harrison_by_rbase(input_csv):

    df = pd.read_csv(input_csv)

    df["gotItRight"] = df["gotItRight"].astype(int)
    df["delta"] = np.abs(df["rbase"] - df["rv"])

    rbase_values = sorted(df["rbase"].unique())

    results = []

    for r in rbase_values:

        subset = df[df["rbase"] == r].copy()

        if len(subset) < 10:
            print(f"Skipping Harrison r={r}, too few samples")
            continue

        x = subset["delta"].values
        y = subset["gotItRight"].values

        p0 = [np.log(0.2), np.log(2.0)]

        try:
            res = minimize(
                neg_ll,
                p0,
                args=(x, y),
                method="L-BFGS-B",
                bounds=[(np.log(1e-4), np.log(1.0)),
                        (np.log(0.5), np.log(3.0))]
            )

            if not res.success:
                print(f"Fit failed for Harrison r={r}")
                continue

            alpha = np.exp(res.x[0])
            beta  = np.exp(res.x[1])
            jnd   = compute_jnd(alpha, beta)

            if jnd > 1:
                print(f"Bad Harrison fit r={r}, JND={jnd:.3f}")
                continue

        except Exception:
            print(f"Error Harrison r={r}")
            continue

        results.append({
            "rbase": r,
            "alpha_h": alpha,
            "beta_h": beta,
            "jnd_h": jnd
        })

    return pd.DataFrame(results)


def summarize_model(df_model_fits):

    summary = (
        df_model_fits
        .groupby("rbase")
        .agg({
            "alpha": "first",
            "beta": "first",
            "jnd": "first"
        })
        .reset_index()
        .rename(columns={
            "alpha": "alpha_m",
            "beta": "beta_m",
            "jnd": "jnd_m"
        })
    )

    return summary


def compute_harrison_ci(df_participant):

    # one JND per participant per rbase per approach
    grouped = (
        df_participant
        .groupby(["participant", "rbase", "approach"])
        .agg({
            "calculatedJND": "median",
            "jndFromHarrison": "median"
        })
        .reset_index()
    )

    print(grouped.head())

    results_calc = []
    results_harr = []

    # ── loop over rbase + approach ─────────────────────────
    for (r, approach), subset in grouped.groupby(["rbase", "approach"]):

        harr_vals = subset["jndFromHarrison"].values
        calc_vals = subset["calculatedJND"].values

        # key fix: filter based on Harrison
        mask = (harr_vals < 0.45) & (calc_vals < 0.45)

        jnds_harr = harr_vals[mask]
        jnds_calc = calc_vals[mask]

        n = len(jnds_harr)

        if n >= 2:

            # ===== calculated =====
            mean_calc = np.mean(jnds_calc)
            std_calc  = np.std(jnds_calc, ddof=1)
            se_calc   = std_calc / np.sqrt(n)

            results_calc.append({
                "rbase": r,
                "approach": approach,
                "mean_jnd": mean_calc,
                "se": se_calc,
                "ci_lower": mean_calc - 1.96 * se_calc,
                "ci_upper": mean_calc + 1.96 * se_calc,
                "n": n
            })

            # ===== Harrison =====
            mean_harr = np.mean(jnds_harr)
            std_harr  = np.std(jnds_harr, ddof=1)
            se_harr   = std_harr / np.sqrt(n)

            results_harr.append({
                "rbase": r,
                "approach": approach,
                "mean_jnd": mean_harr,
                "se": se_harr,
                "ci_lower": mean_harr - 1.96 * se_harr,
                "ci_upper": mean_harr + 1.96 * se_harr,
                "n": n
            })

    df_calc = pd.DataFrame(results_calc)
    df_harr = pd.DataFrame(results_harr)

    return df_calc, df_harr


def compute_model_ci(df_model):

    results = []

    for r, subset in df_model.groupby("rbase"):

        jnd = subset["jnd"].iloc[0]

        # approximate variability from accuracy signal
        acc = subset["jnd"].values
        n = len(acc)

        if n < 2:
            continue

        std = np.std(acc, ddof=1)
        se  = std / np.sqrt(n)

        # propagate rough uncertainty to JND scale
        # (simple approximation)
        ci_lower = jnd - 1.96 * se
        ci_upper = jnd + 1.96 * se

        results.append({
            "rbase": r,
            "mean_jnd": jnd,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper
        })

    return pd.DataFrame(results)


# ── run ───────────────────────────────────────────────────
# INPUT_CSV  = "harrison_results/scatter_positive.csv"
# OUTPUT_CSV = "harrison_results/scatter_positive_weibull_fits_per_participant.csv"

# INPUT_CSV  = "harrison_results/scatter_negative.csv"
# OUTPUT_CSV = "harrison_results/scatter_negative_weibull_fits_per_participant.csv"

# INPUT_CSV  = "harrison_results/parallelCoordinates_positive.csv"
# OUTPUT_CSV = "harrison_results/parallelCoordinates_positive_weibull_fits_per_participant.csv"

INPUT_CSV  = "harrison_results/parallelCoordinates_negative.csv"
OUTPUT_CSV = "harrison_results/parallelCoordinates_negative_weibull_fits_per_participant.csv"




# INPUT_CSV_MODEL = "test_pair_results.csv"
# OUTPUT_CSV_MODEL = "test_pair_weibull_fits.csv"

# INPUT_CSV_MODEL = "test_pair_results_negative.csv"
# OUTPUT_CSV_MODEL = "test_pair_weibull_fits_negative.csv"

# INPUT_CSV_MODEL = "test_pair_results_pcp.csv"
# OUTPUT_CSV_MODEL = "test_pair_weibull_fits_pcp.csv"

INPUT_CSV_MODEL = "test_pair_results_pcp_negative.csv"
OUTPUT_CSV_MODEL = "test_pair_weibull_fits_pcp_negative.csv"




# INPUT_CSV_LOG_REG_MODEL = "test_pair_results_log_reg.csv"
# OUTPUT_CSV_LOG_REG_MODEL = "test_pair_weibull_fits_log_reg.csv"




# INPUT_CSV_SVM_MODEL = "test_pair_results_svm.csv"
# OUTPUT_CSV_SVM_MODEL = "test_pair_weibull_fits_svm.csv"

# INPUT_CSV_SVM_MODEL = "test_pair_results_svm_negative.csv"
# OUTPUT_CSV_SVM_MODEL = "test_pair_weibull_fits_svm_negative.csv"

# INPUT_CSV_SVM_MODEL = "test_pair_results_svm_pcp.csv"
# OUTPUT_CSV_SVM_MODEL = "test_pair_weibull_fits_svm_pcp.csv"

INPUT_CSV_SVM_MODEL = "test_pair_results_svm_pcp_negative.csv"
OUTPUT_CSV_SVM_MODEL = "test_pair_weibull_fits_svm_pcp_negative.csv"

df_result, df_diff = fit_all_participants(INPUT_CSV, OUTPUT_CSV)

df_lda_model_fits = fit_model_by_rbase(INPUT_CSV_MODEL, OUTPUT_CSV_MODEL)
# df_log_reg_model_fits = fit_model_by_rbase(INPUT_CSV_LOG_REG_MODEL, OUTPUT_CSV_LOG_REG_MODEL)
df_svm_model_fits = fit_model_by_rbase(INPUT_CSV_SVM_MODEL, OUTPUT_CSV_SVM_MODEL)

harrison_summary = fit_harrison_by_rbase(INPUT_CSV)
lda_model_summary = summarize_model(df_lda_model_fits)

comparison = pd.merge(harrison_summary, lda_model_summary, on="rbase")

# add difference
comparison["jnd_raw_diff"] = comparison["jnd_m"] - comparison["jnd_h"]
comparison["jnd_abs_diff"] = np.abs(comparison["jnd_m"] - comparison["jnd_h"])

print("\n─────── JND COMPARISON (Harrison vs Model) ────────────\n")
print(comparison.to_string(index=False, float_format="%.4f"))


calc_ci, harrison_ci = compute_harrison_ci(df_result)

harrison_ci_above = harrison_ci[harrison_ci["approach"] == "above"]
harrison_ci_below = harrison_ci[harrison_ci["approach"] == "below"]

calc_ci_above = calc_ci[calc_ci["approach"] == "above"]
calc_ci_below = calc_ci[calc_ci["approach"] == "below"]

harrison_ci_above["ci_plus"]  = harrison_ci_above["ci_upper"] - harrison_ci_above["mean_jnd"]
harrison_ci_above["ci_minus"] = harrison_ci_above["mean_jnd"] - harrison_ci_above["ci_lower"]

calc_ci_above["ci_plus"]  = calc_ci_above["ci_upper"] - calc_ci_above["mean_jnd"]
calc_ci_above["ci_minus"] = calc_ci_above["mean_jnd"] - calc_ci_above["ci_lower"]

harrison_ci_below["ci_plus"]  = harrison_ci_below["ci_upper"] - harrison_ci_below["mean_jnd"]
harrison_ci_below["ci_minus"] = harrison_ci_below["mean_jnd"] - harrison_ci_below["ci_lower"]

calc_ci_below["ci_plus"]  = calc_ci_below["ci_upper"] - calc_ci_below["mean_jnd"]
calc_ci_below["ci_minus"] = calc_ci_below["mean_jnd"] - calc_ci_below["ci_lower"]

lda_model_ci    = compute_model_ci(df_lda_model_fits)
svm_model_ci = compute_model_ci(df_svm_model_fits)

print("\nCalculated JND CI:\n", calc_ci)
print("Harrison CI:\n", harrison_ci)
print("LDA Model CI:\n", lda_model_ci)
print("SVM Model CI:\n", svm_model_ci)



# ── load your fitted CSV ───────────────────────────────
# df = pd.read_csv("harrison_results/scatter_positive_weibull_fits_per_participant.csv")
# df = pd.read_csv("harrison_results/scatter_negative_weibull_fits_per_participant.csv")
# df = pd.read_csv("harrison_results/parallelCoordinates_positive_weibull_fits_per_participant.csv")
df = pd.read_csv("harrison_results/parallelCoordinates_negative_weibull_fits_per_participant.csv")
harrison_agg = fit_harrison_by_rbase(INPUT_CSV)


# ── create subplots ────────────────────────────────────
fig = make_subplots(
    rows=7, cols=1,
    subplot_titles=(
        "Approach: above",
        "Approach: below",
        "Aggregate (Harrison vs Model)",
        "JND with 95% CI (Above)",
        "JND with 95% CI (Below)",
        "JND Difference per Participant",
        "JND Comparison (Harrison vs LDA vs SVM)"
    )
)

# color by rbase
r_vals = sorted(df["rbase"].unique())
colors = px.colors.sequential.Blues

color_map = {
    r: colors[int(i * (len(colors)-1) / max(len(r_vals)-1,1))]
    for i, r in enumerate(r_vals)
}

color_map_model = {
    0.3: "pink",
    0.4: "tomato",
    0.5: "green",
    0.6: "yellow",
    0.7: "orange",
    0.8: "black"
}

# ── group properly ─────────────────────────────────────
grouped = df.groupby(["participant", "rbase", "approach"])

participant_traces = {}
trace_index = 0

grouped = df.groupby(["participant", "rbase", "approach"])

for (participant, rbase, approach), subset in grouped:

    row = 1 if approach == "above" else 2
    color = color_map[rbase]

    alpha = subset["alpha"].iloc[0]
    beta  = subset["beta"].iloc[0]
    jnd   = subset["calculatedJND"].iloc[0]

    x_range = np.linspace(0, subset["delta"].max() + 0.05, 200)
    y_curve = weibull(x_range, alpha, beta)

    name = f"P={participant}, r={rbase}"

    # initialize participant entry
    if participant not in participant_traces:
        participant_traces[participant] = []

    # ── CURVE ──────────────────────────
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=y_curve,
            mode="lines",
            line=dict(color=color),
            name=participant,
            legendgroup=participant,
            showlegend=False,
            hovertemplate=(
                f"Participant: {participant}<br>"
                f"rbase: {rbase}<br>"
                f"approach: {approach}<br>"
                "Δ: %{x:.3f}<br>"
                "P(correct): %{y:.3f}<extra></extra>"
            )
        ),
        row=row, col=1
    )
    participant_traces[participant].append(trace_index)
    trace_index += 1

    # ── DATA ───────────────────────────
    edges = np.linspace(0, subset["delta"].max() + 0.01, 8)
    centers, accs = [], []

    for k in range(len(edges)-1):
        bin_data = subset[
            (subset["delta"] >= edges[k]) &
            (subset["delta"] < edges[k+1])
        ]
        if len(bin_data) >= 5:
            centers.append((edges[k] + edges[k+1]) / 2)
            accs.append(bin_data["gotItRight"].mean())

    fig.add_trace(
        go.Scatter(
            x=centers,
            y=accs,
            mode="markers",
            marker=dict(color=color, size=6),
            legendgroup=participant,
            showlegend=False
        ),
        row=row, col=1
    )
    participant_traces[participant].append(trace_index)
    trace_index += 1

    # ── JND ────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=[jnd],
            y=[0.75],
            mode="markers",
            marker=dict(color=color, size=10, symbol="diamond"),
            legendgroup=participant,
            showlegend=False
        ),
        row=row, col=1
    )
    participant_traces[participant].append(trace_index)
    trace_index += 1

lda_model_grouped = df_lda_model_fits.groupby("rbase")

for r, subset in lda_model_grouped:

    alpha = subset["alpha"].iloc[0]
    beta  = subset["beta"].iloc[0]
    jnd   = subset["jnd"].iloc[0]

    color_model = color_map_model[-r] # change here!!!!

    x_range = np.linspace(0, subset["delta_corr"].max() + 0.05, 200)
    y_curve = weibull(x_range, alpha, beta)

    # curve
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=y_curve,
            mode="lines",
            line=dict(color=color_model, width=3),
            name=f"Model r={r}",
            legendgroup=f"model_{r}",
            showlegend=True,
            hovertemplate=(
                f"MODEL<br>"
                f"rbase: {r}<br>"
                f"α={alpha:.3f}<br>"
                f"β={beta:.3f}<br>"
                "Δ: %{x:.3f}<br>"
                "P: %{y:.3f}<extra></extra>"
            )
        ),
        row=1, col=1
    )

    # duplicate on second subplot
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=y_curve,
            mode="lines",
            line=dict(color=color_model, width=3),
            legendgroup=f"model_{r}",
            showlegend=False
        ),
        row=2, col=1
    )

    # JND marker
    for row_i in [1, 2]:
        fig.add_trace(
            go.Scatter(
                x=[jnd],
                y=[0.75],
                mode="markers",
                marker=dict(
                    color=color_model,
                    size=14,
                    symbol="diamond",
                    line=dict(color="black", width=1)
                ),
                legendgroup=f"model_{r}",
                showlegend=False,
                hovertemplate=(
                    f"MODEL JND<br>"
                    f"rbase: {r}<br>"
                    f"JND: {jnd:.3f}<extra></extra>"
                )
            ),
            row=row_i, col=1
        )


for _, row in harrison_agg.iterrows():

    r = row["rbase"]
    alpha = row["alpha_h"]
    beta  = row["beta_h"]
    jnd   = row["jnd_h"]

    color = color_map_model[r]

    x_range = np.linspace(0, 1, 200)
    y_curve = weibull(x_range, alpha, beta)

    # curve
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=y_curve,
            mode="lines",
            line=dict(color=color, dash="solid", width=3),
            name=f"Harrison r={r}",
            legendgroup=f"agg_{r}",
            showlegend=True,
            hovertemplate=(
                f"HARRISON<br>"
                f"rbase: {r}<br>"
                f"α={alpha:.3f}<br>"
                f"β={beta:.3f}<br>"
                "Δ: %{x:.3f}<br>"
                "P: %{y:.3f}<extra></extra>"
            )
        ),
        row=3, col=1
    )

    # JND marker (circle)
    fig.add_trace(
        go.Scatter(
            x=[jnd],
            y=[0.75],
            mode="markers",
            marker=dict(
                color=color,
                size=12,
                symbol="circle",
                line=dict(color="black", width=1)
            ),
            legendgroup=f"agg_{r}",
            showlegend=False,
            hovertemplate=(
                f"HARRISON JND<br>"
                f"rbase: {r}<br>"
                f"JND: {jnd:.3f}<extra></extra>"
            )
        ),
        row=3, col=1
    )
    

for _, row in lda_model_summary.iterrows():

    r = row["rbase"]
    alpha = row["alpha_m"]
    beta  = row["beta_m"]
    jnd   = row["jnd_m"]

    color = color_map_model[-r] # change here!!!!

    x_range = np.linspace(0, 1, 200)
    y_curve = weibull(x_range, alpha, beta)

    # curve (dashed so it's distinguishable)
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=y_curve,
            mode="lines",
            line=dict(color=color, dash="dash", width=3),
            name=f"Model r={r}",
            legendgroup=f"agg_{r}",
            showlegend=True,
            hovertemplate=(
                f"MODEL<br>"
                f"rbase: {r}<br>"
                f"α={alpha:.3f}<br>"
                f"β={beta:.3f}<br>"
                "Δ: %{x:.3f}<br>"
                "P: %{y:.3f}<extra></extra>"
            )
        ),
        row=3, col=1
    )

    # JND marker (diamond)
    fig.add_trace(
        go.Scatter(
            x=[jnd],
            y=[0.75],
            mode="markers",
            marker=dict(
                color=color,
                size=14,
                symbol="diamond",
                line=dict(color="black", width=1)
            ),
            legendgroup=f"agg_{r}",
            showlegend=False,
            hovertemplate=(
                f"MODEL JND<br>"
                f"rbase: {r}<br>"
                f"JND: {jnd:.3f}<extra></extra>"
            )
        ),
        row=3, col=1
    )

offset = 0.015

fig.add_trace(
    go.Scatter(
        x=harrison_ci_above["rbase"] - offset,
        y=harrison_ci_above["mean_jnd"],
        mode="markers",
        marker=dict(
            color="blue",
            size=8,
            symbol="circle"
        ),
        customdata=np.stack(
            (
                harrison_ci_above["rbase"],
                harrison_ci_above["ci_plus"],
                harrison_ci_above["ci_minus"]
            ),
            axis=-1
        ),
        hovertemplate=(
            "rbase: %{customdata[0]:.2f}<br>"
            "Mean JND: %{y:.4f}<br>"
            "CI: +%{customdata[1]:.4f} / -%{customdata[2]:.4f}<extra></extra>"
        ),
        error_y=dict(
            type="data",
            symmetric=False,
            array=harrison_ci_above["ci_plus"],
            arrayminus=harrison_ci_above["ci_minus"],
            # array=harrison_ci_above["se"],
            # arrayminus=harrison_ci_above["se"],
        ),
        name="Harrison (Above)"
    ),
    row=4, col=1
)

fig.add_trace(
    go.Scatter(
        x=calc_ci_above["rbase"] + offset,
        y=calc_ci_above["mean_jnd"],
        mode="markers",
        marker=dict(
            color="red",
            size=8,
            symbol="diamond"
        ),
        customdata=np.stack(
            (
                calc_ci_above["rbase"],
                calc_ci_above["ci_plus"],
                calc_ci_above["ci_minus"]
            ),
            axis=-1
        ),
        hovertemplate=(
            "rbase: %{customdata[0]:.2f}<br>"
            "Mean JND: %{y:.4f}<br>"
            "CI: +%{customdata[1]:.4f} / -%{customdata[2]:.4f}<extra></extra>"
        ),
        error_y=dict(
            type="data",
            symmetric=False,
            array=calc_ci_above["ci_plus"],
            arrayminus=calc_ci_above["ci_minus"],
            # array=calc_ci_above["se"],
            # arrayminus=calc_ci_above["se"],
        ),
        name="Calculated (Above)"
    ),
    row=4, col=1
)

# Harrison BELOW
fig.add_trace(
    go.Scatter(
        x=harrison_ci_below["rbase"] - offset,
        y=harrison_ci_below["mean_jnd"],
        mode="markers",
        marker=dict(color="blue", size=8, symbol="circle"),
        customdata=np.stack(
            (
                harrison_ci_below["rbase"],
                harrison_ci_below["ci_plus"],
                harrison_ci_below["ci_minus"]
            ),
            axis=-1
        ),
        hovertemplate=(
            "rbase: %{customdata[0]:.2f}<br>"
            "Mean JND: %{y:.4f}<br>"
            "CI: +%{customdata[1]:.4f} / -%{customdata[2]:.4f}<extra></extra>"
        ),
        error_y=dict(
            type="data",
            symmetric=False,
            array=harrison_ci_below["ci_plus"],
            arrayminus=harrison_ci_below["ci_minus"],
            # array=harrison_ci_below["se"],
            # arrayminus=harrison_ci_below["se"],
        ),
        name="Harrison (Below)"
    ),
    row=5, col=1
)

# Calculated BELOW
fig.add_trace(
    go.Scatter(
        x=calc_ci_below["rbase"] + offset,
        y=calc_ci_below["mean_jnd"],
        mode="markers",
        marker=dict(color="red", size=8, symbol="diamond"),
        customdata=np.stack(
            (
                calc_ci_below["rbase"],
                calc_ci_below["ci_plus"],
                calc_ci_below["ci_minus"]
            ),
            axis=-1
        ),
        hovertemplate=(
            "rbase: %{customdata[0]:.2f}<br>"
            "Mean JND: %{y:.4f}<br>"
            "CI: +%{customdata[1]:.4f} / -%{customdata[2]:.4f}<extra></extra>"
        ),
        error_y=dict(
            type="data",
            symmetric=False,
            array=calc_ci_below["ci_plus"],
            arrayminus=calc_ci_below["ci_minus"],
            # array=calc_ci_below["se"],
            # arrayminus=calc_ci_below["se"],

        ),
        name="Calculated (Below)"
    ),
    row=5, col=1
)


fig.add_trace(
    go.Scatter(
        x=df_diff["participant_idx"],
        y=df_diff["jnd_diff"],
        mode="markers",
        marker=dict(
            size=10,
            color="black",
        ),
        text=df_diff["participant"],

        customdata=np.stack(
            (df_diff["rbase"], df_diff["approach"]),
            axis=-1
        ),

        hovertemplate=(
            "Participant: %{text}<br>"
            "rbase: %{customdata[0]}<br>"
            "approach: %{customdata[1]}<br>"
            "JND Difference: %{y:.4f}<extra></extra>"
        ),

        name="JND Difference"
    ),
    row=6, col=1
)

offset_two = 0.015
fig.add_trace(
    go.Scatter(
        x=harrison_ci_above["rbase"] - 2 * offset_two,
        y=harrison_ci_above["mean_jnd"],
        mode="markers",
        marker=dict(
            color="red",
            size=8,
            symbol="diamond"
        ),
        customdata=np.stack(
            (
                harrison_ci_above["rbase"],
                harrison_ci_above["ci_plus"],
                harrison_ci_above["ci_minus"]
            ),
            axis=-1
        ),
        hovertemplate=(
            "rbase: %{customdata[0]:.2f}<br>"
            "Mean JND: %{y:.4f}<br>"
            "CI: +%{customdata[1]:.4f} / -%{customdata[2]:.4f}<extra></extra>"
        ),
        error_y=dict(
            type="data",
            symmetric=False,
            array=harrison_ci_above["ci_plus"],
            arrayminus=harrison_ci_above["ci_minus"],
        ),
        name="Harrison (Above)"
    ),
    row=7, col=1
)

fig.add_trace(
    go.Scatter(
        x=harrison_ci_below["rbase"] - offset_two,
        y=harrison_ci_below["mean_jnd"],
        mode="markers",
        marker=dict(
            color="orange",
            size=8,
            symbol="diamond"
        ),
        customdata=np.stack(
            (
                harrison_ci_below["rbase"],
                harrison_ci_below["ci_plus"],
                harrison_ci_below["ci_minus"]
            ),
            axis=-1
        ),
        hovertemplate=(
            "rbase: %{customdata[0]:.2f}<br>"
            "Mean JND: %{y:.4f}<br>"
            "CI: +%{customdata[1]:.4f} / -%{customdata[2]:.4f}<extra></extra>"
        ),
        error_y=dict(
            type="data",
            symmetric=False,
            array=harrison_ci_below["ci_plus"],
            arrayminus=harrison_ci_below["ci_minus"],
        ),
        name="Harrison (Below)"
    ),
    row=7, col=1
)

fig.add_trace(
    go.Scatter(
        x=-lda_model_ci["rbase"], # change here!!!!
        y=lda_model_ci["mean_jnd"],
        mode="markers",
        marker=dict(
            color="green",
            size=8,
            symbol="circle"
        ),
        customdata=np.stack(
            (
                lda_model_ci["rbase"],
            ),
            axis=-1
        ),
        hovertemplate=(
            "rbase: %{customdata[0]:.2f}<br>"
            "Mean JND: %{y:.4f}<br>"
        ),
        # error_y=dict(
        #     type="data",
        #     symmetric=False,
        #     array=calc_ci_above["ci_plus"],
        #     arrayminus=calc_ci_above["ci_minus"],
        # ),
        name="LDA Model"
    ),
    row=7, col=1
)

fig.add_trace(
    go.Scatter(
        x=-svm_model_ci["rbase"] + offset_two, # change here!!!!
        y=svm_model_ci["mean_jnd"],
        mode="markers",
        marker=dict(
            color="blue",
            size=8,
            symbol="circle"
        ),
        customdata=np.stack(
            (
                svm_model_ci["rbase"],
            ),
            axis=-1
        ),
        hovertemplate=(
            "rbase: %{customdata[0]:.2f}<br>"
            "Mean JND: %{y:.4f}<br>"
        ),
        # error_y=dict(
        #     type="data",
        #     symmetric=False,
        #     array=calc_ci_above["ci_plus"],
        #     arrayminus=calc_ci_above["ci_minus"],
        # ),
        name="SVM Model"
    ),
    row=7, col=1
)


buttons = []

all_traces = list(range(trace_index))

# show all
buttons.append(dict(
    label="All",
    method="update",
    args=[{"visible": [True] * trace_index}]
))

# one button per participant
for p, trace_ids in participant_traces.items():

    visible = [False] * trace_index
    for idx in trace_ids:
        visible[idx] = True

    buttons.append(dict(
        label=p,
        method="update",
        args=[{"visible": visible}]
    ))

# ── reference lines ────────────────────────────────────
for r in [1, 2, 3]:
    fig.add_hline(y=0.75, line_dash="dot", row=r, col=1)
    fig.add_hline(y=0.5, line_dash="dash", row=r, col=1)

fig.add_hline(y=0, line_dash="dash", row=5, col=1)

# ── layout ─────────────────────────────────────────────
fig.update_layout(
    height=2800,   # or 2000–3000 depending on how big you want it
    width=900,     # optional but helps make plots less squeezed
    margin=dict(l=60, r=40, t=80, b=60),
    updatemenus=[
        dict(
            buttons=buttons,
            direction="down",
            showactive=True,
            x=1.05,
            y=1,
        )
    ]
)

# fig.update_xaxes(title_text="|Δ Correlation|")
# fig.update_yaxes(title_text="P(correct)")

fig.update_xaxes(range=[0, 1], title_text="|Δ Correlation|")
fig.update_yaxes(range=[0.5, 1], title_text="P(correct)")

fig.update_xaxes(
    tickvals=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    title_text="rbase",
    row=4, col=1
)
fig.update_yaxes(range=[0, 0.28], title_text="JND", row=4, col=1)


fig.update_xaxes(
    tickvals=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    title_text="rbase",
    row=5, col=1
)
fig.update_yaxes(range=[0, 0.28], title_text="JND", row=5, col=1)


fig.update_xaxes(
    range=[0, len(df_diff) + 0.5],  # centers points nicely
    row=6, col=1
)
fig.update_yaxes(
    range=[-1, 0.5],
    title_text="JND Difference (Harrison - Our Fit)",
    row=6, col=1
)

fig.update_xaxes(
    tickvals=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    title_text="rbase",
    row=7, col=1
)
fig.update_yaxes(range=[0, 0.35], title_text="JND", row=7, col=1)

# fig.write_html("psychometric_plot.html", auto_open=False)
# print("Saved to psychometric_plot.html") 

# fig.write_html("psychometric_plot_negative.html", auto_open=False)
# print("Saved to psychometric_plot_negative.html") 

# fig.write_html("psychometric_plot_pcp.html", auto_open=False)
# print("Saved to psychometric_plot_pcp.html") 

fig.write_html("psychometric_plot_negative_pcp.html", auto_open=False)
print("Saved to psychometric_plot_negative_pcp.html") 
