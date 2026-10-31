"""Validate and register the tourism dataset."""

import argparse
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger("visit_with_us.register")


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
        "--output", default="artifacts/registration"
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
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise RuntimeError(
            f"Unable to load tourism dataset: {exc}"
        ) from exc

    required = {target, customer_id}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Required dataset columns missing: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError("Dataset contains no customer records.")

    if df.columns.duplicated().any():
        raise ValueError("Duplicate column names detected.")

    if df[customer_id].isna().any():
        raise ValueError("Missing customer identifiers detected.")

    if df[customer_id].duplicated().any():
        raise ValueError("Duplicate customer identifiers detected.")

    if df[target].isna().any():
        raise ValueError("Missing purchase outcomes detected.")

    if set(df[target].unique()) != {0, 1}:
        raise ValueError(
            "ProdTaken must contain binary classes 0 and 1."
        )

    # Validate feature eligibility against the CSV schema.
    for model_name, model_config in config["models"].items():
        excluded = set(model_config["excluded_columns"])
        features = [
            col for col in df.columns
            if col not in excluded
            and col not in {target, customer_id}
        ]

        if not features:
            raise ValueError(
                f"{model_name}: no eligible features remain."
            )

    output_dir.mkdir(parents=True, exist_ok=True)

    checksum = sha256_file(source)

    registration = {
        "dataset": source.name,
        "sha256": checksum,
        "version": f"sha256-{checksum[:16]}",
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "target": target,
        "purchase_rate": float(df[target].mean()),
        "registered_at_utc": datetime.now(
            timezone.utc
        ).isoformat()
    }

    report_path = output_dir / "registration.json"
    report_path.write_text(
        json.dumps(registration, indent=2),
        encoding="utf-8"
    )

    LOGGER.info(
        "Dataset registered: %d records, SHA256=%s",
        len(df), checksum
    )

    print(json.dumps(registration, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        LOGGER.exception("Dataset registration failed.")
        raise
