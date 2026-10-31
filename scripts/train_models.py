"""Train and validate Visit with Us classification pipelines."""

import argparse
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

LOGGER = logging.getLogger("visit_with_us.training")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_pipeline(features, estimator):
    numeric = features.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical = [
        col for col in features.columns
        if col not in numeric
    ]

    transformers = []

    if numeric:
        transformers.append((
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            numeric
        ))

    if categorical:
        transformers.append((
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(
                    strategy="most_frequent"
                )),
                ("encoder", OneHotEncoder(
                    handle_unknown="ignore"
                ))
            ]),
            categorical
        ))

    if not transformers:
        raise ValueError("No supported model features.")

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", estimator)
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prepared", default="artifacts/prepared"
    )
    parser.add_argument(
        "--config", default="config/pipeline.json"
    )
    parser.add_argument(
        "--output", default="artifacts/training"
    )
    parser.add_argument(
        "--iterations", type=int, default=8
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

    prepared_dir = Path(args.prepared)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads(
        Path(args.config).read_text(encoding="utf-8")
    )

    target = config["dataset"]["target"]
    random_state = int(config["training"]["random_state"])

    y_path = prepared_dir / "train_target.csv"

    if not y_path.is_file():
        raise FileNotFoundError(
            f"Training labels missing: {y_path}"
        )

    y_train = pd.read_csv(y_path)[target]

    if set(y_train.unique()) != {0, 1}:
        raise ValueError("Training target must contain 0 and 1.")

    if y_train.value_counts().min() < 5:
        raise ValueError(
            "Insufficient minority-class records for 5-fold CV."
        )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state
    )

    scoring = {
        "average_precision": "average_precision",
        "roc_auc": "roc_auc",
        "f1": "f1"
    }

    algorithms = {
        "DummyClassifier": DummyClassifier(
            strategy="prior"
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=1500,
            class_weight="balanced",
            random_state=random_state
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            min_samples_leaf=3,
            random_state=random_state,
            n_jobs=1
        )
    }

    tuning_spaces = {
        "LogisticRegression": {
            "classifier__C": [0.01, 0.1, 1, 10],
            "classifier__class_weight": [
                None, "balanced"
            ]
        },
        "RandomForest": {
            "classifier__n_estimators": [80, 120, 160],
            "classifier__max_depth": [5, 10, None],
            "classifier__min_samples_leaf": [1, 3, 5]
        }
    }

    experiments = []
    champions = {}

    for track in ["model_a", "model_b"]:
        LOGGER.info("Training model track: %s", track)

        feature_path = (
            prepared_dir / track / "train_features.csv"
        )

        if not feature_path.is_file():
            raise FileNotFoundError(
                f"Training features missing: {feature_path}"
            )

        X_train = pd.read_csv(feature_path)

        if len(X_train) != len(y_train):
            raise ValueError(
                f"{track}: feature/target row count mismatch."
            )

        forbidden = {
            target,
            config["dataset"]["customer_id"]
        }

        if forbidden.intersection(X_train.columns):
            raise ValueError(
                f"{track}: target or identifier leakage."
            )

        track_results = []
        candidate_pipelines = {}

        # Baseline and candidate cross-validation.
        for algorithm_name, estimator in algorithms.items():

            pipeline = build_pipeline(
                X_train,
                clone(estimator)
            )

            cv_result = cross_validate(
                pipeline,
                X_train,
                y_train,
                scoring=scoring,
                cv=cv,
                n_jobs=2,
                error_score="raise"
            )

            result = {
                "track": track,
                "algorithm": algorithm_name,
                "variant": "baseline",
                "average_precision": float(
                    np.mean(cv_result["test_average_precision"])
                ),
                "roc_auc": float(
                    np.mean(cv_result["test_roc_auc"])
                ),
                "f1": float(
                    np.mean(cv_result["test_f1"])
                )
            }

            experiments.append(result)
            track_results.append(result)

            candidate_pipelines[algorithm_name] = pipeline

            LOGGER.info(
                "%s %s AP=%.4f",
                track,
                algorithm_name,
                result["average_precision"]
            )

        # Tune the best non-dummy candidate.
        eligible = [
            result for result in track_results
            if result["algorithm"] != "DummyClassifier"
        ]

        best_candidate = max(
            eligible,
            key=lambda item: item["average_precision"]
        )

        algorithm_name = best_candidate["algorithm"]

        search = RandomizedSearchCV(
            estimator=clone(
                candidate_pipelines[algorithm_name]
            ),
            param_distributions=tuning_spaces[algorithm_name],
            n_iter=args.iterations,
            scoring="average_precision",
            cv=cv,
            random_state=random_state,
            n_jobs=2,
            error_score="raise",
            refit=True
        )

        search.fit(X_train, y_train)

        tuned_result = {
            "track": track,
            "algorithm": algorithm_name,
            "variant": "tuned",
            "average_precision": float(search.best_score_),
            "best_parameters": search.best_params_
        }

        experiments.append(tuned_result)

        # Compare tuning candidates using development CV.
        # These scores are not unbiased generalization estimates.
        best_result = max(
            track_results + [tuned_result],
            key=lambda item: item["average_precision"]
        )

        if best_result["variant"] == "tuned":
            champion = search.best_estimator_
        else:
            champion = clone(
                candidate_pipelines[best_result["algorithm"]]
            )
            champion.fit(X_train, y_train)

        model_path = (
            output_dir / f"{track}_champion.joblib"
        )

        joblib.dump(champion, model_path)

        # Smoke-test fitted model on training records only.
        sample = X_train.iloc[:5]
        probabilities = champion.predict_proba(sample)[:, 1]

        if (
            not np.all(np.isfinite(probabilities))
            or np.any(probabilities < 0)
            or np.any(probabilities > 1)
        ):
            raise RuntimeError(
                f"{track}: invalid champion probabilities."
            )

        champions[track] = {
            "algorithm": best_result["algorithm"],
            "variant": best_result["variant"],
            "cv_average_precision": float(
                best_result["average_precision"]
            ),
            "feature_names": X_train.columns.tolist(),
            "artifact": str(model_path),
            "sha256": sha256_file(model_path)
        }

    # Verify both champion tracks completed.
    if set(champions) != {"model_a", "model_b"}:
        raise RuntimeError(
            "Both champion models were not generated."
        )

    # Quality gate: models must exceed a dummy baseline
    # based on training prevalence.
    baseline_ap = float(y_train.mean())

    for track, metadata in champions.items():
        if metadata["cv_average_precision"] <= baseline_ap:
            raise RuntimeError(
                f"{track}: champion did not exceed "
                "training prevalence baseline."
            )

    experiments_path = output_dir / "experiments.json"
    experiments_path.write_text(
        json.dumps(experiments, indent=2, default=str),
        encoding="utf-8"
    )

    manifest = {
        "project": config["project"]["name"],
        "champions": champions,
        "training_records": int(len(y_train)),
        "dataset_version": config["project"]["version"],
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat()
    }

    manifest_path = output_dir / "champion_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, default=str),
        encoding="utf-8"
    )

    LOGGER.info(
        "Training complete: %d experiments, %d champions.",
        len(experiments),
        len(champions)
    )

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        LOGGER.exception("Automated training failed.")
        raise
