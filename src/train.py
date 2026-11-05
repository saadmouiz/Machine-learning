"""Entraînement et comparaison des classifieurs classiques."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config import CLASS_TO_IDX, CLASSES, MODELS_DIR, RANDOM_STATE, TEST_SIZE, VAL_SIZE
from src.dataset import load_image_paths
from src.features import extract_many


def build_models():
    return {
        "k-NN": KNeighborsClassifier(n_neighbors=5, weights="distance"),
        "SVM": SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced", probability=True),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=4,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def make_pipeline(clf) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=0.95, random_state=RANDOM_STATE)),
            ("clf", clf),
        ]
    )


def metrics_dict(y_true, y_pred, labels):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "report": classification_report(y_true, y_pred, labels=labels, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def _group_holdout(indices, y, groups, test_size, random_state):
    """Choose a group-safe holdout close to the requested class proportions."""
    subset_y = y[indices]
    subset_groups = groups[indices]
    splitter = GroupShuffleSplit(n_splits=128, test_size=test_size, random_state=random_state)
    class_values = np.unique(subset_y)
    best = None
    best_score = float("inf")

    for keep_local, hold_local in splitter.split(indices, subset_y, groups=subset_groups):
        keep_y, hold_y = subset_y[keep_local], subset_y[hold_local]
        if not np.all(np.isin(class_values, keep_y)) or not np.all(np.isin(class_values, hold_y)):
            continue
        class_errors = [
            abs(np.count_nonzero(hold_y == label) / np.count_nonzero(subset_y == label) - test_size)
            for label in class_values
        ]
        size_error = abs(len(hold_local) / len(indices) - test_size)
        score = float(np.mean(class_errors) + size_error)
        if score < best_score:
            best_score = score
            best = (indices[keep_local], indices[hold_local])

    if best is None:
        raise ValueError(
            "Impossible de créer des splits groupés contenant toutes les classes. "
            "Il faut davantage d’images sources par classe."
        )
    return best


def make_split_indices(paths, y):
    """Split stratifié approximativement 70/15/15 sans partager une image source."""
    groups = np.asarray([Path(str(path)).stem.split("__", 1)[0] for path in paths])
    all_indices = np.arange(len(y))
    train_val_indices, test_indices = _group_holdout(
        all_indices, y, groups, TEST_SIZE, RANDOM_STATE
    )
    train_indices, val_indices = _group_holdout(
        train_val_indices,
        y,
        groups,
        VAL_SIZE / (1.0 - TEST_SIZE),
        RANDOM_STATE + 1,
    )
    return train_indices, val_indices, test_indices


def train_all():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    paths, labels = load_image_paths()
    y = np.asarray([CLASS_TO_IDX[label] for label in labels], dtype=np.int64)
    X = extract_many(paths)

    train_idx, val_idx, test_idx = make_split_indices(paths, y)
    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
    p_train, p_val, p_test = paths[train_idx], paths[val_idx], paths[test_idx]

    np.savez(
        MODELS_DIR / "splits.npz",
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )
    joblib.dump({"train": p_train, "val": p_val, "test": p_test}, MODELS_DIR / "split_paths.joblib")

    results = {}
    fitted = {}
    for name, clf in build_models().items():
        print(f"\n=== Entraînement : {name} ===")
        pipe = make_pipeline(clf)
        pipe.fit(X_train, y_train)
        y_val_pred = pipe.predict(X_val)
        y_test_pred = pipe.predict(X_test)
        results[name] = {
            "val": metrics_dict(y_val, y_val_pred, labels=list(range(len(CLASSES)))),
            "test": metrics_dict(y_test, y_test_pred, labels=list(range(len(CLASSES)))),
        }
        print(f"Validation  acc={results[name]['val']['accuracy']:.3f}  f1={results[name]['val']['f1_macro']:.3f}")
        print(f"Test        acc={results[name]['test']['accuracy']:.3f}  f1={results[name]['test']['f1_macro']:.3f}")
        fitted[name] = pipe
        joblib.dump(pipe, MODELS_DIR / f"{name.replace(' ', '_').lower()}.joblib")

    best_name = max(results, key=lambda n: results[n]["val"]["f1_macro"])
    final_model = make_pipeline(build_models()[best_name])
    final_model.fit(np.concatenate([X_train, X_val]), np.concatenate([y_train, y_val]))
    joblib.dump(final_model, MODELS_DIR / "best_model.joblib")
    summary = {
        "classes": CLASSES,
        "best_model": best_name,
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "feature_dim": int(X.shape[1]),
        "results": {
            name: {
                "val": {k: v for k, v in m["val"].items() if k != "report"},
                "test": {k: v for k, v in m["test"].items() if k != "report"},
                "val_report": m["val"]["report"],
                "test_report": m["test"]["report"],
            }
            for name, m in results.items()
        },
    }
    (MODELS_DIR / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nMeilleur modèle (F1 validation) : {best_name}")
    return summary


def main():
    train_all()


if __name__ == "__main__":
    main()
