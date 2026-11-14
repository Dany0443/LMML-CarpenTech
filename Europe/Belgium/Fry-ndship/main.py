from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

fields = pd.read_csv(ROOT / "data" / "potato_fields.csv", encoding="utf-8")
fields["short_side_m"] = 10_000 * fields["area_ha"] / fields["long_side_m"]
edge = fields["edge_m"]
train, test = train_test_split(fields.index, test_size=0.2, random_state=42)

features = {
    "Original baseline": ["area_ha", "long_side_m", "ditch_m"],
    "Dimensions only": ["long_side_m", "short_side_m"],
    "Declared measurements": ["long_side_m", "short_side_m", "ditch_m"],
    # on-site hedge measurements are unavailable for unsurveyed fields
    "On-site hedge diagnostic": [
        "long_side_m", "short_side_m", "ditch_m", "hedge_m"
    ],
}

fits = {}
scores = []
for name, cols in features.items():
    dimensions = fields[cols]
    model = LinearRegression().fit(dimensions.loc[train], edge.loc[train])
    edge_fit = pd.Series(model.predict(dimensions), index=fields.index)
    edge_gap = edge - edge_fit
    fits[name] = (edge_fit, edge_gap)
    scores.append({
        "model": name,
        "train_r2": r2_score(edge.loc[train], edge_fit.loc[train]),
        "train_mae_m": mean_absolute_error(edge.loc[train], edge_fit.loc[train]),
        "test_r2": r2_score(edge.loc[test], edge_fit.loc[test]),
        "test_mae_m": mean_absolute_error(edge.loc[test], edge_fit.loc[test]),
    })
    print(f"\n{name}")
    print(pd.Series(model.coef_, index=cols).to_string())
    print(f"intercept: {model.intercept_:.6f}")

print(f"\nFields: {len(fields)}; train: {len(train)}; test: {len(test)}")
print(pd.DataFrame(scores).to_string(index=False, float_format="%.8f"))

fig, axes = plt.subplots(4, 1, figsize=(18, 12), layout="constrained")
for ax, (name, (edge_fit, edge_gap)) in zip(axes, fits.items()):
    ax.scatter(edge_fit, edge_gap, s=4, color="black")
    ax.set(title=name, xlabel="Predicted edge (m)", ylabel="Observed − predicted (m)")
fig.savefig(FIGURES / "model_comparison.png", dpi=170)
plt.close(fig)

edge_fit, edge_gap = fits["Declared measurements"]
rng = np.random.default_rng(42)
fig, axes = plt.subplots(3, 1, figsize=(18, 10), layout="constrained")
for ax, rows, title in [
    (axes[0], fields.index, "Declared measurements: all surveyed fields (exploration)"),
    (axes[1], test, "Same training fit: held-out fields only"),
]:
    ax.scatter(edge_fit.loc[rows], edge_gap.loc[rows], s=5, color="black")
    ax.set(title=title, xlabel="Predicted edge (m)", ylabel="Observed − predicted (m)")

axes[2].scatter(edge_fit, rng.permutation(edge_gap.to_numpy()), s=5, color="black")
axes[2].set(
    title="Control: residuals randomly reassigned to predicted edges",
    xlabel="Predicted edge (m)",
    ylabel="Shuffled residual (m)",
)
fig.savefig(FIGURES / "survey_evidence.png", dpi=170)
plt.close(fig)

print(f"\nFigures saved in {FIGURES}")
