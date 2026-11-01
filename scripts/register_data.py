"""Register and validate the public tourism dataset."""

import argparse
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger("visit_with_us.registration")


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
        default="artifacts/registration"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

    source = Path(args.data)
    config_path = Path(args.config)
    output_dir = Path(args.output)

    if not source.is_file() or source.stat().st_size == 0:
        raise FileNotFoundError(
            f"Dataset missing or empty: {source}"
        )

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Pipeline configuration missing: {config_path}"
        )

    config = json.loads(
        config_path.read_text(encoding="utf-8")
    )

    dataset_config = config["dataset"]

    target = dataset_config["target"]
    customer_id = dataset_config["customer_id"]

    try:
        df = pd.read_csv(source)

    except (
        OSError,
        ValueError,
        pd.errors.ParserError
    ) as exc:
        raise RuntimeError(
            "Unable to load the tourism dataset."
        ) from exc

    if df.empty:
        raise ValueError("Dataset contains no records.")

    if df.columns.duplicated().any():
        raise ValueError(
            "Duplicate dataset column names detected."
        )

    missing_columns = (
        {target, customer_id} - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Required dataset columns missing: "
            f"{sorted(missing_columns)}"
        )

    if df[customer_id].isna().any():
        raise ValueError(
            "Missing customer identifiers detected."
        )

    if df[customer_id].duplicated().any():
        raise ValueError(
            "Duplicate customer identifiers detected."
        )

    if df[target].isna().any():
        raise ValueError(
            "Missing target values detected."
        )

    if set(df[target].unique()) != {0, 1}:
        raise ValueError(
            "Target must contain binary classes 0 and 1."
        )

    feature_counts = {}

    for track, model_config in config["models"].items():

        excluded = (
            set(model_config["excluded_columns"])
            | {target, customer_id}
        )

        eligible_features = [
            column
            for column in df.columns
            if column not in excluded
        ]

        if not eligible_features:
            raise ValueError(
                f"{track}: no eligible features remain."
            )

        feature_counts[track] = len(
            eligible_features
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    checksum = sha256_file(source)

    registration = {
        "dataset": source.name,
        "sha256": checksum,
        "dataset_version": f"sha256-{checksum[:16]}",
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "target": target,
        "purchase_rate": float(df[target].mean()),
        "feature_counts": feature_counts,
        "registered_at_utc": datetime.now(
            timezone.utc
        ).isoformat()
    }

    report_path = (
        output_dir / "registration.json"
    )

    report_path.write_text(
        json.dumps(registration, indent=2),
        encoding="utf-8"
    )

    LOGGER.info(
        "Dataset registered successfully: %d records.",
        len(df)
    )

    print(json.dumps(registration, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        LOGGER.exception(
            "Dataset registration failed."
        )
        raise
