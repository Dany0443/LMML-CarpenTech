from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (
    GroupShuffleSplit,
    LeaveOneGroupOut,
    LeaveOneOut,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

here = Path(__file__).resolve().parent
evidence_dir = here / "evidence"
evidence_dir.mkdir(exist_ok=True)

shipped_seed = 1137
sweep_seeds = range(200)
identity_cols = ["submission_id", "date_of_birth", "citizenship"]

apps = pd.read_csv(here / "applications.csv")
apps["person_id"] = apps.groupby(["date_of_birth", "citizenship"]).ngroup()
feature_cols = [c for c in apps.columns if c not in identity_cols + ["label", "person_id"]]
X = apps[feature_cols].to_numpy()
y = apps["label"].to_numpy()
person = apps["person_id"].to_numpy()
X_prescaled = StandardScaler().fit_transform(X)


def new_model():
    return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=1))


def knn_only():
    return KNeighborsClassifier(n_neighbors=1)


def score_split(model, features, train_idx, test_idx):
    model.fit(features[train_idx], y[train_idx])
    return accuracy_score(y[test_idx], model.predict(features[test_idx]))


def shipped_holdout(seed):
    train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=0.2, random_state=seed)
    return score_split(knn_only(), X_prescaled, train_idx, test_idx)


def scaler_fixed_holdout(seed):
    train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=0.2, random_state=seed)
    return score_split(new_model(), X, train_idx, test_idx)


def group_holdout_leaky_scaler(seed):
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, test_idx = next(splitter.split(X, y, person))
    return score_split(knn_only(), X_prescaled, train_idx, test_idx)


def group_holdout_clean(seed):
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, test_idx = next(splitter.split(X, y, person))
    return score_split(new_model(), X, train_idx, test_idx)


def sweep(fn):
    return np.array([fn(seed) for seed in sweep_seeds])


def oof_accuracy(model, features, cv, groups=None):
    preds = cross_val_predict(model, features, y, cv=cv, groups=groups)
    return accuracy_score(y, preds), preds


def describe(name, scores):
    return {
        "variant": name,
        "mean_acc": scores.mean(),
        "sd_acc": scores.std(ddof=1),
        "min_acc": scores.min(),
        "max_acc": scores.max(),
    }


print(f"rows: {len(apps)}  distinct people (dob+citizenship): {apps.person_id.nunique()}")
apps_per_person = apps.groupby("person_id").size()
print("applications per person:", apps_per_person.value_counts().sort_index().to_dict())
mixed = apps.groupby("person_id")["label"].nunique().gt(1).sum()
print(f"people with conflicting labels: {mixed}")
print(f"negative form_fill_seconds rows: {(apps.form_fill_seconds < 0).sum()}")

# twin check: within-person vs cross-person nearest distance in scaled space
dist = pairwise_distances(X_prescaled)
np.fill_diagonal(dist, np.inf)
same_person = person[:, None] == person[None, :]
within_nn = np.where(same_person, dist, np.inf).min(axis=1)
across_nn = np.where(~same_person, dist, np.inf).min(axis=1)
has_twin = np.isfinite(within_nn)
print(
    f"nearest same-person twin: median {np.median(within_nn[has_twin]):.4f}, max {within_nn[has_twin].max():.4f}"
)
print(
    f"nearest other person:     median {np.median(across_nn):.4f}, min {across_nn.min():.4f}"
)
print(f"rows with a twin: {has_twin.sum()} ({has_twin.mean():.1%})")

shipped_acc = shipped_holdout(shipped_seed)
print(f"\nshipped pipeline, seed {shipped_seed}: {shipped_acc:.4f}")

shipped_scores = sweep(shipped_holdout)
scaler_scores = sweep(scaler_fixed_holdout)
group_leaky_scores = sweep(group_holdout_leaky_scaler)
group_clean_scores = sweep(group_holdout_clean)

sweep_table = pd.DataFrame(
    {
        "seed": list(sweep_seeds),
        "shipped": shipped_scores,
        "fix_scaler_only": scaler_scores,
        "fix_groups_only": group_leaky_scores,
        "fix_scaler_and_groups": group_clean_scores,
    }
)
sweep_table.to_csv(evidence_dir / "seed_sweep.csv", index=False)

# cv fix applied alone: shipped (pre-scaled) features, no groups
cv10_acc, _ = oof_accuracy(knn_only(), X_prescaled, StratifiedKFold(10, shuffle=True, random_state=0))
loo_row_acc, loo_row_preds = oof_accuracy(knn_only(), X_prescaled, LeaveOneOut())
loo_row_clean_acc, _ = oof_accuracy(new_model(), X, LeaveOneOut())

# real fix: leave one person out, scaler refit inside each fold
final_acc, final_preds = oof_accuracy(new_model(), X, LeaveOneGroupOut(), groups=person)
final_correct = int((final_preds == y).sum())
final_leaky_acc, _ = oof_accuracy(knn_only(), X_prescaled, LeaveOneGroupOut(), groups=person)

dob_only = apps.groupby("date_of_birth").ngroup().to_numpy()
dob_only_acc, _ = oof_accuracy(new_model(), X, LeaveOneGroupOut(), groups=dob_only)

clipped = X.copy()
clipped[:, feature_cols.index("form_fill_seconds")] = np.clip(
    clipped[:, feature_cols.index("form_fill_seconds")], 0, None
)
clipped_acc, _ = oof_accuracy(new_model(), clipped, LeaveOneGroupOut(), groups=person)

summary_rows = [
    describe("shipped: leaky scaler + row hold-out", shipped_scores),
    describe("decoy 1 fixed: scaler fit on train only", scaler_scores),
    describe("real bug fixed: person-grouped hold-out (leaky scaler kept)", group_leaky_scores),
    describe("both fixed: person-grouped hold-out + clean scaler", group_clean_scores),
]
summary = pd.DataFrame(summary_rows)
summary.to_csv(evidence_dir / "variant_summary.csv", index=False)
print("\n200-seed 80/20 hold-out sweep")
print(summary.round(4).to_string(index=False))

paired_scaler = scaler_scores - shipped_scores
print(
    f"\ndecoy 1 paired diff (fixed - shipped): mean {paired_scaler.mean():+.4f}, "
    f"sd {paired_scaler.std(ddof=1):.4f}, identical on {(paired_scaler == 0).mean():.0%} of seeds"
)
within_band = abs(shipped_acc - shipped_scores.mean()) <= 2 * shipped_scores.std(ddof=1)
print(
    f"decoy 2: seed {shipped_seed} scored {shipped_acc:.4f}; 200-seed mean {shipped_scores.mean():.4f} "
    f"(sd {shipped_scores.std(ddof=1):.4f}); within 2 sd: {within_band}"
)
print(f"decoy 2 fixed with 10-fold row CV: {cv10_acc:.4f}; row-level LOO: {loo_row_acc:.4f}")
print(f"row-level LOO with scaler inside folds: {loo_row_clean_acc:.4f}")

print(f"\nleave-one-person-out, scaler inside folds: {final_acc:.4f} ({final_correct}/{len(y)})")
print(f"leave-one-person-out, leaky scaler kept:   {final_leaky_acc:.4f}")
print(f"grouping by date_of_birth alone:           {dob_only_acc:.4f}")
print(f"negative form_fill_seconds clipped to 0:   {clipped_acc:.4f}")

flag = f"SIGMOID_{round(final_acc * 100)}"
print(f"\nFLAG: {flag}")

oof_table = apps[["submission_id", "person_id", "label"]].copy()
oof_table["pred_loo_row"] = loo_row_preds
oof_table["pred_leave_one_person_out"] = final_preds
oof_table["correct_leave_one_person_out"] = (final_preds == y).astype(int)
oof_table.to_csv(evidence_dir / "oof_predictions.csv", index=False)
apps.to_csv(evidence_dir / "applications_with_person_id.csv", index=False)

(evidence_dir / "final_result.txt").write_text(
    f"shipped_reported_acc={shipped_acc:.4f}\n"
    f"true_acc_leave_one_person_out={final_acc:.4f}\n"
    f"correct={final_correct}/{len(y)}\n"
    f"flag={flag}\n"
)
