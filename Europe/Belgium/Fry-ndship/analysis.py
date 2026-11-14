from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

fields = pd.read_csv(ROOT / "data" / "potato_fields.csv", encoding="utf-8")

print(fields.head())
print("\nDataset shape:", fields.shape)
print("\nColumns:", fields.columns.tolist())
print("\nMissing values:")
print(fields.isnull().sum())
print("\nField measurements:")
print(fields[["area_ha", "long_side_m", "ditch_m", "hedge_m", "edge_m"]].describe().round(2))

columns = ["area_ha", "long_side_m", "ditch_m", "hedge_m", "edge_m"]

print("\nCorrelations with edge length:")
print(fields[columns].corr()["edge_m"].sort_values(ascending=False).round(3))

plt.scatter(fields["ditch_m"], fields["edge_m"], s=5, alpha=0.4)

plt.xlabel("Ditch length (m)")
plt.ylabel("Edge length (m)")
plt.title("Ditch length vs edge length")

plt.savefig(
    FIGURES / "ditch_vs_edge.png",
    dpi=150,
    bbox_inches="tight"
)
plt.close()

plt.figure(figsize=(10, 6))
plt.scatter(fields["lon"], fields["lat"], s=4, alpha=0.5)

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Potato field locations")
plt.savefig(
    FIGURES / "field_locations.png",
    dpi=150,
    bbox_inches="tight"
)
plt.close()

short_side = fields["area_ha"] * 10000 / fields["long_side_m"]
rect_edge = 2 * (fields["long_side_m"] + short_side)

edge_gap = fields["edge_m"] - rect_edge

print("\nDifference from rectangular perimeter:")
print(edge_gap.describe().round(2))

fields["edge_gap"] = edge_gap

print("\nCorrelations with perimeter difference:")
print(
    fields[["edge_gap", "hedge_m", "ditch_m", "area_ha", "long_side_m"]]
    .corr()["edge_gap"]
    .round(3)
)

relative_gap = 100 * edge_gap / rect_edge

print("\nRelative perimeter difference (%):")
print(relative_gap.describe().round(2))

plt.figure(figsize=(12, 6))
plt.scatter(fields["ditch_m"], fields["hedge_m"], s=4, alpha=0.4)

plt.xlabel("Ditch length (m)")
plt.ylabel("Hedge length (m)")
plt.title("Ditch vs hedge measurements")

plt.savefig(
    FIGURES / "ditch_hedge.png",
    dpi=180,
    bbox_inches="tight"
)
plt.close()

slope, intercept = np.polyfit(fields["ditch_m"], fields["hedge_m"], 1)
hedge_gap = fields["hedge_m"] - (slope * fields["ditch_m"] + intercept)

print("\nHedge residual statistics:")
print(hedge_gap.describe().round(3))

plt.figure(figsize=(16, 5))
plt.scatter(fields["ditch_m"], hedge_gap, s=4, alpha=0.6)

plt.xlabel("Ditch length (m)")
plt.ylabel("Hedge residual (m)")
plt.title("Hedge measurements after removing linear trend")

plt.savefig(
    FIGURES / "hedge_residuals.png",
    dpi=200,
    bbox_inches="tight"
)
plt.close()

print("\nFirst 15 fields:")
print(fields[["field_id", "ditch_m", "hedge_m", "edge_m"]].head(15).to_string(index=False))

print("\nFields by crop and farm type:")
print(pd.crosstab(fields["crop_code"], fields["farm_type"]))

survey_cols = ["field_id", "crop_code", "crop", "farm_type"]

print("\nSurvey order:")
print(fields[survey_cols].head(40).to_string(index=False))
print("\nUnique values:")
print(fields[survey_cols].nunique())

for col in ["crop_code", "farm_type"]:
    run_id = fields[col].ne(fields[col].shift()).cumsum()
    runs = fields.groupby(run_id, sort=False).agg(
        label=(col, "first"),
        rows=(col, "size"),
    )
    print(f"\n{col}: {len(runs)} consecutive runs")
    print(runs.head(30).to_string(index=False))
