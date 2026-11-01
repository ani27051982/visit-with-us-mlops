"""Train and evaluate Visit with Us champion classifiers."""

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
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

LOGGER = logging.getLogger("visit_with_us.training")


def sha256_file(path):
    """Calculate artifact SHA-256 checksum."""

    digest = hashlib.sha256()

    with Path(path).open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1024 * 1024),
            b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def build_pipeline(features, estimator):
    """Build a leakage-safe fitted preprocessing pipeline."""

    numeric_columns = features.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_columns = [
        column
        for column in features.columns
        if column not in numeric_columns
    ]

    transformers = []

    if numeric_columns:
        transformers.append((
            "numeric",
            Pipeline([
                (
                    "imputer",
                    SimpleImputer(strategy="median")
                ),
                (
                    "scaler",
                    StandardScaler()
                )
            ]),
            numeric_columns
        ))

    if categorical_columns:
        transformers.append((
            "categorical",
            Pipeline([
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    )
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    )
                )
            ]),
            categorical_columns
        ))

    if not transformers:
        raise ValueError(
            "No eligible numeric or categorical features."
        )

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
        "--prepared",
        default="artifacts/prepared"
    )

    parser.add_argument(
        "--config",
        default="config/pipeline.json"
    )

    parser.add_argument(
        "--output",
        default="artifacts/training"
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=8
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

    if args.iterations < 1:
        raise ValueError(
            "Hyperparameter iterations must be positive."
        )

    prepared_dir = Path(args.prepared)
    config_path = Path(args.config)
    output_dir = Path(args.output)

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Training configuration missing: {config_path}"
        )

    config = json.loads(
        config_path.read_text(encoding="utf-8")
    )

    target = config["dataset"]["target"]

    random_state = int(
        config["training"]["random_state"]
    )

    cv_folds = int(
        config["training"].get("cv_folds", 5)
    )

    if cv_folds < 2:
        raise ValueError(
            "Cross-validation requires at least two folds."
        )

    target_path = (
        prepared_dir / "train_target.csv"
    )

    if not target_path.is_file():
        raise FileNotFoundError(
            "Prepared training target is missing."
        )

    y_train = pd.read_csv(
        target_path
    )[target]

    if y_train.isna().any():
        raise ValueError(
            "Training target contains missing values."
        )

    if set(y_train.unique()) != {0, 1}:
        raise ValueError(
            "Training target must contain classes 0 and 1."
        )

    if y_train.value_counts().min() < cv_folds:
        raise ValueError(
            "Insufficient minority-class samples for CV."
        )

    cv = StratifiedKFold(
        n_splits=cv_folds,
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
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=1
        )
    }

    tuning_spaces = {
        "LogisticRegression": {
            "classifier__C": [
                0.01, 0.1, 1.0, 10.0
            ],
            "classifier__class_weight": [
                None, "balanced"
            ]
        },
        "RandomForest": {
            "classifier__n_estimators": [
                80, 120, 160
            ],
            "classifier__max_depth": [
                5, 10, None
            ],
            "classifier__min_samples_leaf": [
                1, 3, 5
            ]
        }
    }

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    experiments = []
    champions = {}

    # --------------------------------------------------------
    # Train Model A and Model B independently.
    # --------------------------------------------------------

    for track in ["model_a", "model_b"]:

        LOGGER.info(
            "Starting training for %s.", track
        )

        feature_path = (
            prepared_dir
            / track
            / "train_features.csv"
        )

        if not feature_path.is_file():
            raise FileNotFoundError(
                f"{track}: prepared features missing."
            )

        X_train = pd.read_csv(
            feature_path
        )

        if X_train.empty:
            raise ValueError(
                f"{track}: training features are empty."
            )

        if len(X_train) != len(y_train):
            raise ValueError(
                f"{track}: feature/target row mismatch."
            )

        forbidden_columns = {
            target,
            config["dataset"]["customer_id"]
        }

        if forbidden_columns.intersection(
            X_train.columns
        ):
            raise ValueError(
                f"{track}: target or identifier leakage."
            )

        candidate_results = []
        candidate_pipelines = {}

        # ----------------------------------------------------
        # Baseline model comparison using stratified CV.
        # ----------------------------------------------------

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
                    np.mean(
                        cv_result[
                            "test_average_precision"
                        ]
                    )
                ),
                "roc_auc": float(
                    np.mean(
                        cv_result["test_roc_auc"]
                    )
                ),
                "f1": float(
                    np.mean(
                        cv_result["test_f1"]
                    )
                )
            }

            experiments.append(result)
            candidate_results.append(result)

            candidate_pipelines[
                algorithm_name
            ] = pipeline

            LOGGER.info(
                "%s | %s | AP=%.4f",
                track,
                algorithm_name,
                result["average_precision"]
            )

        # ----------------------------------------------------
        # Select best non-dummy algorithm for tuning.
        # ----------------------------------------------------

        eligible_candidates = [
            result
            for result in candidate_results
            if result["algorithm"] != "DummyClassifier"
        ]

        best_candidate = max(
            eligible_candidates,
            key=lambda item: item["average_precision"]
        )

        selected_algorithm = (
            best_candidate["algorithm"]
        )

        search = RandomizedSearchCV(
            estimator=clone(
                candidate_pipelines[
                    selected_algorithm
                ]
            ),
            param_distributions=tuning_spaces[
                selected_algorithm
            ],
            n_iter=args.iterations,
            scoring="average_precision",
            cv=cv,
            random_state=random_state,
            n_jobs=2,
            error_score="raise",
            refit=True
        )

        search.fit(
            X_train,
            y_train
        )

        tuned_result = {
            "track": track,
            "algorithm": selected_algorithm,
            "variant": "tuned",
            "average_precision": float(
                search.best_score_
            ),
            "best_parameters": search.best_params_
        }

        experiments.append(
            tuned_result
        )

        # ----------------------------------------------------
        # Select champion from baseline and tuned candidates.
        # ----------------------------------------------------

        best_result = max(
            candidate_results + [tuned_result],
            key=lambda item: item["average_precision"]
        )

        if best_result["variant"] == "tuned":

            champion = search.best_estimator_

        else:

            champion = clone(
                candidate_pipelines[
                    best_result["algorithm"]
                ]
            )

            champion.fit(
                X_train,
                y_train
            )

        # ----------------------------------------------------
        # Serialize champion model.
        # ----------------------------------------------------

        model_path = (
            output_dir
            / f"{track}_champion.joblib"
        )

        joblib.dump(
            champion,
            model_path
        )

        # ----------------------------------------------------
        # Smoke-test champion on training records.
        # ----------------------------------------------------

        sample = X_train.iloc[:5]

        probabilities = champion.predict_proba(
            sample
        )[:, 1]

        if (
            not np.all(
                np.isfinite(probabilities)
            )
            or np.any(probabilities < 0)
            or np.any(probabilities > 1)
        ):
            raise RuntimeError(
                f"{track}: invalid prediction probabilities."
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

        LOGGER.info(
            "Champion selected for %s: %s",
            track,
            best_result["algorithm"]
        )

    # --------------------------------------------------------
    # Final Quality Gate
    # --------------------------------------------------------

    if set(champions) != {
        "model_a", "model_b"
    }:
        raise RuntimeError(
            "Both champion models were not generated."
        )

    baseline_ap = float(
        y_train.mean()
    )

    for track, metadata in champions.items():

        if (
            metadata["cv_average_precision"]
            <= baseline_ap
        ):
            raise RuntimeError(
                f"{track}: champion failed "
                "minimum Average Precision quality gate."
            )

    # --------------------------------------------------------
    # Save Experiment Tracking
    # --------------------------------------------------------

    experiments_path = (
        output_dir / "experiments.json"
    )

    experiments_path.write_text(
        json.dumps(
            experiments,
            indent=2,
            default=str
        ),
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Save Champion Manifest
    # --------------------------------------------------------

    manifest = {
        "project": config["project"]["name"],
        "champions": champions,
        "training_records": int(len(y_train)),
        "dataset_version": config[
            "project"
        ]["version"],
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat()
    }

    manifest_path = (
        output_dir / "champion_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            default=str
        ),
        encoding="utf-8"
    )

    LOGGER.info(
        "Training completed successfully: "
        "%d experiments, %d champions.",
        len(experiments),
        len(champions)
    )

    print(
        json.dumps(
            manifest,
            indent=2
        )
    )


if __name__ == "__main__":

    try:
        main()

    except Exception:
        LOGGER.exception(
            "Automated model training failed."
        )
        raise
