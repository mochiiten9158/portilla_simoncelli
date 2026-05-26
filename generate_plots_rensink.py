import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from scipy.stats import truncnorm


def generate_harrison_scatter(corr=0.0, seed=None, out_dir="assets/harrison_replication", count=0):
    rng = np.random.default_rng(seed)
    n = 100

    def truncated_normal(rng, n):
        a, b = -2, 2
        return truncnorm.rvs(a, b, size=n, random_state=rng)

    x = truncated_normal(rng, n)
    y = truncated_normal(rng, n)

    x = x - x.mean()
    y = y - y.mean()
    y = y * (x.std() / y.std())

    # Step 2 & 3: handle sign of target correlation
    r = corr
    
    if abs(abs(r) - 1.0) < 1e-9:
        y_new = x if r > 0 else (1.0 - x)
    else:
        # For negative target correlations, work with -y so the formula
        # always operates in the "positive mixing" regime, then flip back.
        sign = 1.0 if r >= 0 else -1.0
        y_work = y * sign          # flip y for negative r
        r_work = abs(r)            # work with |r|

        rz = np.corrcoef(x, y_work)[0, 1]   # recompute rz against y_work

        denom = (rz - 1) * (2 * r_work**2 + rz - 1)
        if abs(denom) < 1e-12:
            y_new = r_work * x + np.sqrt(max(0, 1 - r_work**2)) * y_work
        else:
            discriminant = r_work**2 * (rz**2 - 1) * (r_work**2 - 1)
            if discriminant < 0:
                discriminant = 0.0
            lam = ((rz - 1) * (r_work**2 + rz) + np.sqrt(discriminant)) / denom
            y_new = (lam * x + (1 - lam) * y_work) / np.sqrt(lam**2 + (1 - lam)**2)

        # flip the result back so it's negatively correlated with x
        y_new = y_new * sign

    # Step 4: re-normalize to mean=0.5, std=0.2  (following Harrison)
    current_std = x.std()

    x_final = (x / current_std) * 0.2 + 0.5
    y_final = (y_new / current_std) * 0.2 + 0.5

    # Step 5: save 300x300px image with axes (matching Harrison's materials)
    if abs(r) < 1e-6:
        tag = "z"
    elif r > 0:
        tag = "p"
    else:
        tag = "n"

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = out_dir / f"{count}_{r:.4f}.png"

    # 300px at 100 dpi = 3 inches
    fig, ax = plt.subplots(figsize=(3, 3), dpi=100)

    # point size=2 in scatter is in points^2 area units;
    # Harrison specifies "2 pixel" radius points, so s ≈ 4 works
    ax.scatter(x_final, y_final, s=4, color="black", linewidths=0)

    # show axes (Harrison displays left and bottom axes)
    # ax.spines["top"].set_visible(False)
    # ax.spines["right"].set_visible(False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    # ax.set_xticks([])
    # ax.set_yticks([])

    plt.tight_layout(pad=0.1)
    fig.savefig(filename, dpi=100)
    plt.close(fig)

    # print(f"Saved: {filename}  (target r={r:.4f}, achieved r={np.corrcoef(x_final, y_final)[0,1]:.4f})")
    return filename


def main():
    # Harrison tested r = 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 for the staircase.
    # For PS feature extraction across the full range, use the same
    # step size as your existing generator but with the corrected method.

    corrs_positive = [round(i * 0.0025, 4) for i in range(0, 401)]   # 0.0 to 1.0
    corrs_negative = [-round(i * 0.0025, 4) for i in range(0, 401)]  # -0.0025 to -1.0

    # --- generate positive correlations ---
    for r in corrs_positive:
        generate_harrison_scatter(corr=r, seed=None,
                                  out_dir="assets/harrison_positive_100_training")

    # --- generate negative correlations ---
    for r in corrs_negative:
        generate_harrison_scatter(corr=r, seed=None,
                                  out_dir="assets/harrison_negative_100_training")


if __name__ == "__main__":
    main()