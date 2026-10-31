"""Validate trained model artifacts and quality gates."""

import argparse
import hashlib
import json
import logging
from pathlib import Path

LOGGER = logging.getLogger("visit_with_us.validation")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
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
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = training_dir / "champion_manifest.json"

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Champion manifest missing: {manifest_path}"
        )

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    champions = manifest.get("champions", {})

    if set(champions) != {"model_a", "model_b"}:
        raise ValueError(
            "Expected Model A and Model B champion artifacts."
        )

    results = {}

    for track, metadata in champions.items():

        # Resolve the artifact relative to the training
        # directory rather than trusting arbitrary paths.
        artifact_name = Path(metadata["artifact"]).name

        model_path = (
            training_dir / artifact_name
        ).resolve()

        if training_dir not in model_path.parents:
            raise ValueError("Unsafe artifact path.")

        if not model_path.is_file():
            raise FileNotFoundError(
                f"Missing champion model: {model_path}"
            )

        expected_hash = metadata["sha256"]
        actual_hash = sha256_file(model_path)

        if actual_hash != expected_hash:
            raise RuntimeError(
                f"{track}: SHA-256 checksum mismatch."
            )

        ap_score = float(metadata["cv_average_precision"])

        if not 0 <= ap_score <= 1:
            raise ValueError(
                f"{track}: invalid Average Precision."
            )

        if not metadata.get("feature_names"):
            raise ValueError(
                f"{track}: feature metadata missing."
            )

        results[track] = {
            "status": "PASSED",
            "artifact": str(model_path),
            "sha256": actual_hash,
            "cv_average_precision": ap_score
        }

        LOGGER.info(
            "%s validated: AP=%.4f", track, ap_score
        )

    report = {
        "status": "PASSED",
        "models": results,
        "artifact_count": len(results)
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
        LOGGER.exception("Artifact validation failed.")
        raise
