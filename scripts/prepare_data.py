"""Prepare leakage-safe dual-model training datasets."""

import argparse
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

LOGGER = logging.getLogger("visit_with_us.preparation")


def sha256_file(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1024 * 1024),
            b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default="data/raw/tourism.csv"
    )

    parser.add_argument(
        "--config",
        default="config/pipeline.json"
    )

    parser.add_argument(
        "--output",
        default="artifacts/prepared"
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
            f"Dataset missing: {source}"
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
        raise ValueError(
            "Duplicate dataset column names detected."
        )

    for column in [target, customer_id]:
        if column not in df.columns:
            raise ValueError(
                f"Required column missing: {column}"
            )

    if df[customer_id].isna().any():
        raise ValueError(
            "Missing customer identifiers."
        )

    if df[customer_id].duplicated().any():
        raise ValueError(
            "Duplicate customer identifiers."
        )

    if df[target].isna().any():
        raise ValueError(
            "Missing target values."
        )

    if set(df[target].unique()) != {0, 1}:
        raise ValueError(
            "Target must contain binary classes 0 and 1."
        )

    # Normalize categorical text without fitting encoders.
    for column in df.select_dtypes(
        include=["object", "string"]
    ).columns:

        if column != customer_id:
            df[column] = (
                df[column]
                .astype("string")
                .str.strip()
                .replace("", pd.NA)
            )

    test_size = float(
        dataset_config["test_size"]
    )

    random_state = int(
        dataset_config["random_state"]
    )

    if not 0 < test_size < 1:
        raise ValueError(
            "test_size must be between 0 and 1."
        )

    if df[target].value_counts().min() < 2:
        raise ValueError(
            "Insufficient records for stratified splitting."
        )

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[target]
    )

    train_ids = set(
        train_df[customer_id]
    )

    test_ids = set(
        test_df[customer_id]
    )

    if not train_ids.isdisjoint(test_ids):
        raise RuntimeError(
            "Customer leakage detected across splits."
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    artifacts = {}
    feature_counts = {}

    for track, model_config in config["models"].items():

        exclusions = (
            set(model_config["excluded_columns"])
            | {target, customer_id}
        )

        features = [
            column
            for column in df.columns
            if column not in exclusions
        ]

        if not features:
            raise ValueError(
                f"{track}: no eligible features."
            )

        feature_counts[track] = len(features)

        model_dir = (
            output_dir / track
        )

        model_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        for split_name, split_df in [
            ("train", train_df),
            ("test", test_df)
        ]:

            feature_path = (
                model_dir
                / f"{split_name}_features.csv"
            )

            split_df[features].to_csv(
                feature_path,
                index=False
            )

            artifacts[
                f"{track}_{split_name}"
            ] = str(feature_path)

        LOGGER.info(
            "%s prepared with %d features.",
            track,
            len(features)
        )

    for split_name, split_df in [
        ("train", train_df),
        ("test", test_df)
    ]:

        target_path = (
            output_dir
            / f"{split_name}_target.csv"
        )

        split_df[[target]].to_csv(
            target_path,
            index=False
        )

        artifacts[
            f"{split_name}_target"
        ] = str(target_path)

        # Customer identifiers are retained only in
        # private workflow artifacts for traceability.
        ids_path = (
            output_dir
            / f"{split_name}_ids.csv"
        )

        split_df[[customer_id]].to_csv(
            ids_path,
            index=False
        )

        artifacts[
            f"{split_name}_ids"
        ] = str(ids_path)

    checksum = sha256_file(source)

    manifest = {
        "dataset_sha256": checksum,
        "dataset_version": f"sha256-{checksum[:16]}",
        "train_records": int(len(train_df)),
        "test_records": int(len(test_df)),
        "train_purchase_rate": float(
            train_df[target].mean()
        ),
        "test_purchase_rate": float(
            test_df[target].mean()
        ),
        "feature_counts": feature_counts,
        "artifacts": artifacts,
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat()
    }

    manifest_path = (
        output_dir / "prepared_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8"
    )

    LOGGER.info(
        "Data preparation completed: train=%d, test=%d",
        len(train_df),
        len(test_df)
    )

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        LOGGER.exception(
            "Dataset preparation failed."
        )
        raise
