import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import truncnorm


def generate_harrison_pcp(corr=0.0, seed=None, out_dir="assets/harrison_pcp_positive_100", count=0):
    rng = np.random.default_rng(seed)
    n = 100

    def truncated_normal(rng, n):
        a, b = -2, 2
        return truncnorm.rvs(a, b, size=n, random_state=rng)

    x = truncated_normal(rng, n)
    y = truncated_normal(rng, n)

    x = (x - x.mean()) / x.std()
    y = (y - y.mean()) / y.std()

    r = corr
    
    if abs(abs(r) - 1.0) < 1e-9:
        y_new = x if r > 0 else (1.0 - x)
    else:
        sign = 1.0 if r >= 0 else -1.0
        y_work = y * sign
        r_work = abs(r) # work with |r|

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

        y_new = y_new * sign            

    current_std = x.std()

    x_final = (x / current_std) * 0.2 + 0.5
    y_final = (y_new / current_std) * 0.2 + 0.5

    # PCP rendering
    # Two vertical axes at horizontal positions 0 and 1.
    # Each data point i is a line from (0, x_final[i]) to (1, y_final[i]).

    if abs(r) < 1e-6:
        tag = "z"
    elif r > 0:
        tag = "p"
    else:
        tag = "n"

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = out_dir / f"{count}_{r:.4f}.png"

    fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # draw
    for i in range(n):
        ax.plot(
            [0, 1],
            [x_final[i], y_final[i]],
            # color="black",
            linewidth=0.4,
            color="#606060",
            alpha=0.5,
        )

    # draw the two vertical axis lines on top of the data lines as colorless lines
    ax.axvline(x=0, color="none", linewidth=1.2, zorder=5)
    ax.axvline(x=1, color="none", linewidth=1.2, zorder=5)

    ax.set_position([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    plt.tight_layout(pad=0.1)
    fig.savefig(filename, dpi=100, bbox_inches=None, pad_inches=0)
    plt.close(fig)

    # print(f"Saved: {filename}  "
    #       f"(target r={r:.4f}, "
    #       f"achieved r={np.corrcoef(x_final, y_final)[0,1]:.4f})")
    return filename


def main():
    corrs_positive = [round(i * 0.0025, 4) for i in range(0, 401)]
    corrs_negative = [-round(i * 0.0025, 4) for i in range(0, 401)]

    for r in corrs_positive:
        generate_harrison_pcp(corr=r, seed=None,
                               out_dir="assets/harrison_pcp_positive_100_training")

    for r in corrs_negative:
        generate_harrison_pcp(corr=r, seed=None,
                               out_dir="assets/harrison_pcp_negative_100_training")


if __name__ == "__main__":
    main()