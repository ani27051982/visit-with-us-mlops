"""Validate published MLOps champion artifacts."""

import argparse
import hashlib
import json
import logging
import math
import re
from pathlib import Path

LOGGER = logging.getLogger("visit_with_us.validation")


def sha256_file(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1024 * 1024), b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training", default="artifacts/training"
    )
    parser.add_argument(
        "--output", default="artifacts/validation"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    training_dir = Path(args.training).resolve()
    output_dir = Path(args.output).resolve()
    manifest_path = training_dir / "champion_manifest.json"

    if not manifest_path.is_file():
        raise FileNotFoundError(
            "Champion manifest is missing."
        )

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    champions = manifest.get("champions", {})

    if set(champions) != {"model_a", "model_b"}:
        raise ValueError(
            "Both champion tracks are required."
        )

    results = {}

    for track in ("model_a", "model_b"):
        metadata = champions[track]

        model_path = (
            training_dir / f"{track}_champion.joblib"
        ).resolve()

        if model_path.parent != training_dir:
            raise ValueError(
                "Unsafe model artifact path."
            )

        if not model_path.is_file():
            raise FileNotFoundError(
                f"Missing champion artifact: {track}"
            )

        expected = metadata.get("sha256", "")

        if not re.fullmatch(
            r"[0-9a-fA-F]{64}", expected
        ):
            raise ValueError(
                f"{track}: invalid SHA-256 metadata."
            )

        actual = sha256_file(model_path)

        if actual.lower() != expected.lower():
            raise RuntimeError(
                f"{track}: artifact checksum mismatch."
            )

        features = metadata.get("feature_names")

        if (
            not isinstance(features, list)
            or not features
            or len(features) != len(set(features))
        ):
            raise ValueError(
                f"{track}: invalid feature schema."
            )

        score = float(
            metadata["cv_average_precision"]
        )

        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(
                f"{track}: invalid Average Precision."
            )

        results[track] = {
            "status": "PASSED",
            "artifact": model_path.name,
            "sha256": actual,
            "cv_average_precision": score,
            "feature_count": len(features)
        }

        LOGGER.info(
            "%s validated successfully.", track
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "status": "PASSED",
        "artifact_count": len(results),
        "models": results
    }

    (output_dir / "validation_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        LOGGER.exception(
            "Artifact validation failed."
        )
        raise
