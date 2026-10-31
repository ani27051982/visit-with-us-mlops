"""Prepare leakage-safe training and holdout datasets."""

import argparse
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

LOGGER = logging.getLogger("visit_with_us.prepare")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument(
        "--config", default="config/pipeline.json"
    )
    parser.add_argument(
        "--output", default="artifacts/prepared"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

    source = Path(args.data)
    config_path = Path(args.config)
    output_dir = Path(args.output)

    if not source.is_file():
        raise FileNotFoundError(
            f"Source dataset missing: {source}"
        )

    config = json.loads(
        config_path.read_text(encoding="utf-8")
    )

    dataset_config = config["dataset"]
    target = dataset_config["target"]
    customer_id = dataset_config["customer_id"]

    df = pd.read_csv(source)

    if df.empty:
        raise ValueError("Dataset is empty.")

    if df.columns.duplicated().any():
        raise ValueError("Duplicate column names detected.")

    if df[customer_id].isna().any():
        raise ValueError("Missing customer identifiers.")

    if df[customer_id].duplicated().any():
        raise ValueError("Duplicate customer identifiers.")

    if df[target].isna().any():
        raise ValueError("Missing target values.")

    if set(df[target].unique()) != {0, 1}:
        raise ValueError("Target must contain classes 0 and 1.")

    # Normalize categorical text without fitting transforms.
    for column in df.select_dtypes(
        include=["object", "string"]
    ).columns:
        if column != customer_id:
            df[column] = (
                df[column].astype("string").str.strip()
                .replace("", pd.NA)
            )

    test_size = float(dataset_config["test_size"])
    random_state = int(dataset_config["random_state"])

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    if df[target].value_counts().min() < 2:
        raise ValueError(
            "Both target classes need at least two records."
        )

    # Create one shared split for both models.
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[target]
    )

    train_ids = set(train_df[customer_id])
    test_ids = set(test_df[customer_id])

    if not train_ids.isdisjoint(test_ids):
        raise RuntimeError("Customer leakage across splits.")

    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts = {}

    for model_name, model_config in config["models"].items():

        exclusions = (
            set(model_config["excluded_columns"])
            | {target, customer_id}
        )

        features = [
            col for col in df.columns
            if col not in exclusions
        ]

        if not features:
            raise ValueError(
                f"{model_name}: no predictive features."
            )

        model_dir = output_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        for split_name, split_df in [
            ("train", train_df),
            ("test", test_df)
        ]:
            feature_path = (
                model_dir / f"{split_name}_features.csv"
            )

            split_df[features].to_csv(
                feature_path,
                index=False
            )

            artifacts[
                f"{model_name}_{split_name}"
            ] = str(feature_path)

        LOGGER.info(
            "%s prepared with %d features.",
            model_name, len(features)
        )

    # Save shared targets separately.
    for split_name, split_df in [
        ("train", train_df),
        ("test", test_df)
    ]:
        target_path = (
            output_dir / f"{split_name}_target.csv"
        )

        split_df[[target]].to_csv(
            target_path,
            index=False
        )

        artifacts[
            f"{split_name}_target"
        ] = str(target_path)

    # Customer IDs are private traceability artifacts.
    # Do not publish them to public repository branches.
    for split_name, split_df in [
        ("train", train_df),
        ("test", test_df)
    ]:
        ids_path = output_dir / f"{split_name}_ids.csv"
        split_df[[customer_id]].to_csv(
            ids_path, index=False
        )
        artifacts[f"{split_name}_ids"] = str(ids_path)

    manifest = {
        "dataset_sha256": sha256_file(source),
        "dataset_version": (
            f"sha256-{sha256_file(source)[:16]}"
        ),
        "train_records": int(len(train_df)),
        "test_records": int(len(test_df)),
        "train_purchase_rate": float(
            train_df[target].mean()
        ),
        "test_purchase_rate": float(
            test_df[target].mean()
        ),
        "artifacts": artifacts,
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat()
    }

    manifest_path = output_dir / "prepared_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8"
    )

    LOGGER.info(
        "Data preparation complete: train=%d, test=%d",
        len(train_df), len(test_df)
    )

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        LOGGER.exception("Data preparation failed.")
        raise
