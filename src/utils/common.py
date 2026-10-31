
"""
Visit with Us — Enterprise MLOps
Centralized Reusable Utility Library

Shared across:
- Google Colab
- GitHub Actions
- Streamlit
- Standalone Python scripts

All reusable functions are maintained in this module.
"""

import os
import json
import hashlib
import logging
import tempfile

from pathlib import Path
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    log_loss,
    brier_score_loss
)



# ================================================================
# CENTRALIZED UTILITY MODULE
# SECTION 2 — CONFIGURATION, LOGGING & EXCEPTION HANDLING
# ================================================================


# ---------------------------------------------------------------
# Custom Exception Hierarchy
# ---------------------------------------------------------------

class MLOpsError(Exception):
    """Base exception for project-specific pipeline failures."""


class ConfigurationError(MLOpsError):
    """Raised when project configuration is invalid."""


class DatasetValidationError(MLOpsError):
    """Raised when dataset integrity or schema validation fails."""


class ArtifactError(MLOpsError):
    """Raised when artifact persistence or validation fails."""


class ModelEvaluationError(MLOpsError):
    """Raised when model evaluation cannot be completed."""


# ---------------------------------------------------------------
# Logger Access
# ---------------------------------------------------------------

def get_logger(name="visit_with_us_mlops"):
    """
    Return a named project logger.

    Logging handlers are configured centrally by the
    application entry point rather than inside this function.
    """
    return logging.getLogger(name)


def log_and_raise(
    message,
    exception_type=MLOpsError,
    logger=None
):
    """
    Log a meaningful error and raise the requested exception.

    Parameters
    ----------
    message : str
        Human-readable error description.
    exception_type : Exception subclass
        Exception class to raise.
    logger : logging.Logger or None
        Optional logger instance.
    """
    if logger is not None:
        logger.error("%s", message)

    raise exception_type(message)


# ---------------------------------------------------------------
# Directory Management
# ---------------------------------------------------------------

def ensure_directory(path):
    """
    Create a directory and verify it exists.

    Returns
    -------
    pathlib.Path
        Validated directory path.
    """
    directory = Path(path).expanduser()

    try:
        directory.mkdir(
            parents=True,
            exist_ok=True
        )

    except OSError as exc:
        raise ArtifactError(
            f"Unable to create directory '{directory}': {exc}"
        ) from exc

    if not directory.is_dir():
        raise ArtifactError(
            f"Directory validation failed: {directory}"
        )

    return directory


# ---------------------------------------------------------------
# Configuration Loading
# ---------------------------------------------------------------

def load_json(path):
    """
    Load a JSON configuration object safely.

    Raises
    ------
    FileNotFoundError
        When the requested file does not exist.
    ConfigurationError
        When JSON is malformed or not an object.
    """
    file_path = Path(path).expanduser()

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {file_path}"
        )

    try:
        with file_path.open(
            "r",
            encoding="utf-8"
        ) as handle:
            data = json.load(handle)

    except json.JSONDecodeError as exc:
        raise ConfigurationError(
            f"Invalid JSON configuration '{file_path}': "
            f"{exc}"
        ) from exc

    except OSError as exc:
        raise ConfigurationError(
            f"Unable to read configuration '{file_path}': "
            f"{exc}"
        ) from exc

    if not isinstance(data, dict):
        raise ConfigurationError(
            f"Configuration must contain a JSON object: "
            f"{file_path}"
        )

    return data


# ---------------------------------------------------------------
# Configuration Validation
# ---------------------------------------------------------------

def validate_config(config, required_sections=None):
    """
    Validate required project configuration sections.

    Parameters
    ----------
    config : dict
        Loaded project configuration.
    required_sections : iterable or None
        Required top-level configuration keys.

    Returns
    -------
    bool
        True when configuration is valid.
    """
    if not isinstance(config, dict):
        raise ConfigurationError(
            "Project configuration must be a dictionary."
        )

    if not config:
        raise ConfigurationError(
            "Project configuration cannot be empty."
        )

    if required_sections is None:
        required_sections = [
            "project",
            "dataset",
            "models",
            "training",
            "evaluation",
            "logging",
            "deployment"
        ]

    missing_sections = [
        section
        for section in required_sections
        if section not in config
    ]

    if missing_sections:
        raise ConfigurationError(
            "Missing required configuration sections: "
            f"{missing_sections}"
        )

    return True


# ---------------------------------------------------------------
# Atomic JSON Persistence
# ---------------------------------------------------------------

def save_json(data, path):
    """
    Save JSON using a temporary file and atomic replacement.

    This reduces the risk of partially written configuration
    and report files when a write is interrupted.

    Note:
        Atomic replacement depends on filesystem semantics.
        Google Drive-mounted storage may not guarantee the
        same atomicity as a local POSIX filesystem.
    """
    destination = Path(path).expanduser()

    ensure_directory(destination.parent)

    temporary_path = None

    try:
        serialized = json.dumps(
            data,
            indent=4,
            ensure_ascii=False,
            allow_nan=False
        )

        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(destination.parent),
            prefix=".mlops_tmp_",
            suffix=".json",
            delete=False
        ) as handle:

            temporary_path = Path(handle.name)
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(
            str(temporary_path),
            str(destination)
        )

    except (OSError, TypeError, ValueError) as exc:

        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass

        raise ArtifactError(
            f"Unable to save JSON file '{destination}': "
            f"{exc}"
        ) from exc

    return destination


# ---------------------------------------------------------------
# SHA-256 File Integrity
# ---------------------------------------------------------------

def sha256_file(path, chunk_size=1024 * 1024):
    """
    Calculate the SHA-256 checksum of a file.

    Used for:
    - Dataset registration.
    - Model artifact verification.
    - Reproducibility reports.
    - Pipeline artifact integrity.
    """
    file_path = Path(path).expanduser()

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Cannot calculate checksum. "
            f"File does not exist: {file_path}"
        )

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be a positive integer."
        )

    digest = hashlib.sha256()

    try:
        with file_path.open("rb") as handle:

            for chunk in iter(
                lambda: handle.read(chunk_size),
                b""
            ):
                digest.update(chunk)

    except OSError as exc:
        raise ArtifactError(
            f"Unable to calculate SHA-256 for "
            f"'{file_path}': {exc}"
        ) from exc

    return digest.hexdigest()



# ================================================================
# CENTRALIZED UTILITY MODULE
# SECTION 3 — DATASET LOADING & DATA QUALITY
# ================================================================


def load_dataset(path, encoding="utf-8", **read_csv_kwargs):
    """
    Load a CSV dataset with explicit error handling.

    Parameters
    ----------
    path : str or Path
        CSV file location.
    encoding : str
        Character encoding.
    **read_csv_kwargs
        Additional pandas.read_csv options.

    Returns
    -------
    pandas.DataFrame
        Loaded, nonempty dataset.

    Raises
    ------
    FileNotFoundError
        When the file does not exist.
    DatasetValidationError
        When the file is empty, malformed or unreadable.
    """
    file_path = Path(path).expanduser()

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}. "
            "Verify the configured dataset location."
        )

    if file_path.stat().st_size == 0:
        raise DatasetValidationError(
            f"Dataset file is empty: {file_path}"
        )

    try:
        dataframe = pd.read_csv(
            file_path,
            encoding=encoding,
            **read_csv_kwargs
        )

    except (
        OSError,
        UnicodeError,
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
        ValueError
    ) as exc:
        raise DatasetValidationError(
            f"Unable to load dataset '{file_path}': {exc}"
        ) from exc

    if dataframe.empty:
        raise DatasetValidationError(
            f"Dataset contains no records: {file_path}"
        )

    get_logger().info(
        "Dataset loaded successfully: rows=%d, columns=%d",
        dataframe.shape[0],
        dataframe.shape[1]
    )

    return dataframe


def validate_schema(
    dataframe,
    expected_columns,
    target_column,
    allow_extra_columns=False
):
    """
    Validate DataFrame schema against expected columns.

    Detects missing columns, unexpected columns,
    duplicate column names and missing target.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            "Schema validation requires a pandas DataFrame."
        )

    if dataframe.empty:
        raise DatasetValidationError(
            "Schema validation failed: dataset is empty."
        )

    if dataframe.columns.duplicated().any():
        duplicates = dataframe.columns[
            dataframe.columns.duplicated()
        ].tolist()

        raise DatasetValidationError(
            f"Duplicate column names detected: {duplicates}"
        )

    actual_columns = set(dataframe.columns)
    required_columns = set(expected_columns)

    missing_columns = sorted(
        required_columns - actual_columns
    )

    unexpected_columns = sorted(
        actual_columns - required_columns
    )

    if missing_columns:
        raise DatasetValidationError(
            f"Missing required columns: {missing_columns}"
        )

    if unexpected_columns and not allow_extra_columns:
        raise DatasetValidationError(
            f"Unexpected dataset columns: {unexpected_columns}"
        )

    if target_column not in dataframe.columns:
        raise DatasetValidationError(
            f"Target column '{target_column}' is missing."
        )

    get_logger().info(
        "Dataset schema validated: %d columns.",
        len(dataframe.columns)
    )

    return {
        "valid": True,
        "row_count": int(len(dataframe)),
        "column_count": int(dataframe.shape[1]),
        "missing_columns": missing_columns,
        "unexpected_columns": unexpected_columns
    }


def validate_binary_target(
    dataframe,
    target_column,
    expected_classes=(0, 1)
):
    """
    Validate binary target labels and class distribution.

    Both classes must be present for stratified training.
    """
    if target_column not in dataframe.columns:
        raise DatasetValidationError(
            f"Target column not found: {target_column}"
        )

    target = dataframe[target_column]

    if target.isna().any():
        raise DatasetValidationError(
            f"Target '{target_column}' contains missing values."
        )

    actual_classes = set(target.unique().tolist())
    required_classes = set(expected_classes)

    if actual_classes != required_classes:
        raise DatasetValidationError(
            f"Invalid target classes. "
            f"Expected {sorted(required_classes)}, "
            f"found {sorted(actual_classes)}."
        )

    counts = target.value_counts().sort_index()

    get_logger().info(
        "Binary target validated: %s",
        counts.to_dict()
    )

    return {
        "valid": True,
        "class_counts": {
            str(key): int(value)
            for key, value in counts.items()
        },
        "positive_rate": float((target == 1).mean())
    }


def validate_customer_ids(
    dataframe,
    customer_id_column="CustomerID"
):
    """
    Validate customer identifiers for missing and duplicate IDs.

    Customer IDs are identifiers, not predictive features.
    """
    if customer_id_column not in dataframe.columns:
        raise DatasetValidationError(
            f"Customer ID column missing: {customer_id_column}"
        )

    customer_ids = dataframe[customer_id_column]

    missing_count = int(customer_ids.isna().sum())
    duplicate_count = int(customer_ids.duplicated().sum())

    if missing_count:
        raise DatasetValidationError(
            f"{missing_count} missing customer identifiers found."
        )

    if duplicate_count:
        raise DatasetValidationError(
            f"{duplicate_count} duplicate customer IDs found. "
            "Investigate duplicates before train/test splitting."
        )

    return {
        "valid": True,
        "unique_customers": int(customer_ids.nunique()),
        "missing_ids": missing_count,
        "duplicate_ids": duplicate_count
    }


def validate_numeric_ranges(
    dataframe,
    numeric_ranges
):
    """
    Validate configured numerical feature ranges.

    Example
    -------
    numeric_ranges = {
        "Age": (18, 100),
        "MonthlyIncome": (0, None)
    }

    A None boundary indicates no limit.
    """
    if not isinstance(numeric_ranges, dict):
        raise TypeError(
            "numeric_ranges must be a dictionary."
        )

    violations = {}

    for column, bounds in numeric_ranges.items():

        if column not in dataframe.columns:
            raise DatasetValidationError(
                f"Range validation column missing: {column}"
            )

        if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
            raise ValueError(
                f"Invalid range definition for '{column}'. "
                "Expected (minimum, maximum)."
            )

        minimum, maximum = bounds

        values = pd.to_numeric(
            dataframe[column],
            errors="coerce"
        )

        invalid_numeric = (
            dataframe[column].notna() & values.isna()
        )

        nonfinite = (
            values.notna()
            & ~np.isfinite(values)
        )

        out_of_range = pd.Series(
            False,
            index=dataframe.index
        )

        if minimum is not None:
            out_of_range |= values < minimum

        if maximum is not None:
            out_of_range |= values > maximum

        invalid_mask = (
            invalid_numeric | nonfinite | out_of_range
        )

        count = int(invalid_mask.sum())

        if count:
            violations[column] = count

    if violations:
        raise DatasetValidationError(
            "Numeric range validation failed: "
            f"{violations}"
        )

    return {
        "valid": True,
        "validated_columns": list(numeric_ranges),
        "violations": {}
    }


def generate_data_quality_report(
    dataframe,
    target_column=None
):
    """
    Generate a JSON-compatible data-quality report.

    Reports missing values, duplicates, data types,
    unique values and target distribution.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            "Data quality reporting requires a DataFrame."
        )

    if dataframe.empty:
        raise DatasetValidationError(
            "Cannot profile an empty dataset."
        )

    missing_counts = dataframe.isna().sum()

    report = {
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "rows": int(dataframe.shape[0]),
        "columns": int(dataframe.shape[1]),
        "duplicate_rows": int(
            dataframe.duplicated().sum()
        ),
        "missing_values_total": int(
            missing_counts.sum()
        ),
        "missing_by_column": {
            str(column): int(value)
            for column, value in missing_counts.items()
        },
        "missing_percentage_by_column": {
            str(column): round(
                float(value / len(dataframe) * 100),
                2
            )
            for column, value in missing_counts.items()
        },
        "data_types": {
            str(column): str(dtype)
            for column, dtype in dataframe.dtypes.items()
        },
        "unique_values_by_column": {
            str(column): int(dataframe[column].nunique())
            for column in dataframe.columns
        }
    }

    if target_column is not None:
        target_report = validate_binary_target(
            dataframe,
            target_column
        )

        report["target"] = target_report

    get_logger().info(
        "Data quality report generated: rows=%d, "
        "columns=%d, missing=%d, duplicates=%d",
        report["rows"],
        report["columns"],
        report["missing_values_total"],
        report["duplicate_rows"]
    )

    return report



# ================================================================
# CENTRALIZED UTILITY MODULE
# SECTION 4 — DATA SPLITTING & LEAKAGE-SAFE PREPROCESSING
# ================================================================


def create_train_test_split(
    dataframe,
    target_column,
    test_size=0.20,
    random_state=42
):
    """
    Create a reproducible stratified train/test split.

    Both Model A and Model B must use the same split.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Validated customer dataset.
    target_column : str
        Binary classification target.
    test_size : float
        Fraction of records reserved for testing.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    tuple
        X_train, X_test, y_train, y_test
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            "Expected a pandas DataFrame for splitting."
        )

    validate_binary_target(
        dataframe,
        target_column
    )

    if not 0 < test_size < 1:
        raise ValueError(
            "test_size must be strictly between 0 and 1."
        )

    if not isinstance(random_state, int):
        raise TypeError(
            "random_state must be an integer."
        )

    X = dataframe.drop(
        columns=[target_column]
    )

    y = dataframe[target_column]

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

    except ValueError as exc:
        raise DatasetValidationError(
            "Unable to create stratified train/test split. "
            f"Details: {exc}"
        ) from exc

    if set(X_train.index).intersection(set(X_test.index)):
        raise DatasetValidationError(
            "Training and testing indices overlap. "
            "Potential data leakage detected."
        )

    get_logger().info(
        "Stratified split completed: "
        "train=%d, test=%d",
        len(X_train),
        len(X_test)
    )

    return X_train, X_test, y_train, y_test


def select_model_features(
    dataframe,
    excluded_columns,
    strict=False
):
    """
    Create a model-specific feature view.

    Model A and Model B use different exclusion policies.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            "Feature selection requires a DataFrame."
        )

    if dataframe.empty:
        raise DatasetValidationError(
            "Cannot select features from an empty DataFrame."
        )

    excluded_columns = list(excluded_columns)

    missing_exclusions = [
        column
        for column in excluded_columns
        if column not in dataframe.columns
    ]

    if strict and missing_exclusions:
        raise DatasetValidationError(
            "Configured excluded columns not found: "
            f"{missing_exclusions}"
        )

    selected = dataframe.drop(
        columns=excluded_columns,
        errors="raise" if strict else "ignore"
    ).copy()

    if selected.shape[1] == 0:
        raise DatasetValidationError(
            "Feature selection removed all columns."
        )

    return selected


def validate_feature_alignment(
    X_train,
    X_test
):
    """
    Ensure train/test feature schemas match exactly.

    This check must occur before fitting preprocessing.
    """
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError(
            "X_train must be a pandas DataFrame."
        )

    if not isinstance(X_test, pd.DataFrame):
        raise TypeError(
            "X_test must be a pandas DataFrame."
        )

    if X_train.empty or X_test.empty:
        raise DatasetValidationError(
            "Train/test feature datasets cannot be empty."
        )

    train_columns = list(X_train.columns)
    test_columns = list(X_test.columns)

    if train_columns != test_columns:
        raise DatasetValidationError(
            "Train/test feature columns differ. "
            "Ensure identical feature selection and ordering."
        )

    for column in train_columns:
        if X_train[column].dtype != X_test[column].dtype:
            raise DatasetValidationError(
                f"Feature dtype mismatch for '{column}'. "
                f"Train={X_train[column].dtype}, "
                f"Test={X_test[column].dtype}"
            )

    return {
        "valid": True,
        "feature_count": len(train_columns),
        "feature_names": train_columns
    }


def identify_feature_types(X_train):
    """
    Identify numeric and categorical training features.

    Feature types are determined using training data only.
    """
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError(
            "X_train must be a pandas DataFrame."
        )

    if X_train.empty:
        raise DatasetValidationError(
            "Cannot identify feature types from empty data."
        )

    numeric_columns = X_train.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X_train.select_dtypes(
        include=["object", "category", "bool", "string"]
    ).columns.tolist()

    unsupported_columns = sorted(
        set(X_train.columns)
        - set(numeric_columns)
        - set(categorical_columns)
    )

    if unsupported_columns:
        raise DatasetValidationError(
            "Unsupported feature data types detected: "
            f"{unsupported_columns}. "
            "Convert these columns explicitly before training."
        )

    return {
        "numeric": numeric_columns,
        "categorical": categorical_columns
    }


def build_preprocessor(
    X_train,
    numeric_scaling=True,
    numeric_imputation="median",
    categorical_imputation="most_frequent"
):
    """
    Build an unfitted ColumnTransformer.

    The transformer is fitted only inside the ML pipeline
    during model training and cross-validation.

    Numeric features:
        - Missing-value imputation
        - Optional standard scaling

    Categorical features:
        - Missing-value imputation
        - One-hot encoding
        - Unknown-category handling
    """
    feature_types = identify_feature_types(X_train)

    numeric_columns = feature_types["numeric"]
    categorical_columns = feature_types["categorical"]

    if not numeric_columns and not categorical_columns:
        raise DatasetValidationError(
            "No supported predictive features were found."
        )

    numeric_steps = [
        (
            "imputer",
            SimpleImputer(
                strategy=numeric_imputation
            )
        )
    ]

    if numeric_scaling:
        numeric_steps.append(
            (
                "scaler",
                StandardScaler()
            )
        )

    categorical_steps = [
        (
            "imputer",
            SimpleImputer(
                strategy=categorical_imputation
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]

    transformers = []

    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(numeric_steps),
                numeric_columns
            )
        )

    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(categorical_steps),
                categorical_columns
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True
    )

    get_logger().info(
        "Preprocessor created: numeric=%d, categorical=%d",
        len(numeric_columns),
        len(categorical_columns)
    )

    return preprocessor


def build_model_pipeline(
    preprocessor,
    estimator
):
    """
    Combine preprocessing and classification.

    The complete pipeline ensures preprocessing is fitted
    independently within each cross-validation fold.
    """
    if preprocessor is None:
        raise ValueError(
            "Preprocessor cannot be None."
        )

    if estimator is None:
        raise ValueError(
            "Estimator cannot be None."
        )

    if not hasattr(estimator, "fit"):
        raise TypeError(
            "Estimator must implement fit()."
        )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", estimator)
    ])


def prepare_dual_model_features(
    X_train,
    X_test,
    model_config
):
    """
    Prepare independent feature views for Model A and Model B.

    Both models retain identical training/testing records.

    Parameters
    ----------
    X_train, X_test : pandas.DataFrame
        Shared training/testing partitions.
    model_config : dict
        Configuration containing model-specific exclusions.

    Returns
    -------
    dict
        Model-specific training/testing feature DataFrames.
    """
    if not isinstance(model_config, dict):
        raise TypeError(
            "model_config must be a dictionary."
        )

    prepared = {}

    for model_key, details in model_config.items():

        excluded = details.get(
            "excluded_columns",
            []
        )

        train_features = select_model_features(
            X_train,
            excluded_columns=excluded,
            strict=False
        )

        test_features = select_model_features(
            X_test,
            excluded_columns=excluded,
            strict=False
        )

        alignment = validate_feature_alignment(
            train_features,
            test_features
        )

        prepared[model_key] = {
            "X_train": train_features,
            "X_test": test_features,
            "feature_count": alignment["feature_count"],
            "feature_names": alignment["feature_names"]
        }

        get_logger().info(
            "%s features prepared: %d columns",
            model_key,
            alignment["feature_count"]
        )

    return prepared



# ================================================================
# CENTRALIZED UTILITY MODULE
# SECTION 5 — MODEL EVALUATION & PREDICTION
# ================================================================


def get_positive_probabilities(model, X, positive_class=1):
    """
    Extract positive-class probabilities from a fitted model.

    Uses model.classes_ instead of assuming the positive
    class is always in probability column 1.
    """
    if not hasattr(model, "predict_proba"):
        raise ModelEvaluationError(
            "The fitted classifier must support predict_proba()."
        )

    if not hasattr(model, "classes_"):
        raise ModelEvaluationError(
            "Classifier is not fitted or classes_ is unavailable."
        )

    classes = list(model.classes_)

    if positive_class not in classes:
        raise ModelEvaluationError(
            f"Positive class {positive_class!r} is missing. "
            f"Available classes: {classes}"
        )

    try:
        probabilities = np.asarray(
            model.predict_proba(X)
        )

    except (ValueError, TypeError, AttributeError) as exc:
        raise ModelEvaluationError(
            f"Probability prediction failed: {exc}"
        ) from exc

    if (
        probabilities.ndim != 2
        or probabilities.shape[1] != len(classes)
    ):
        raise ModelEvaluationError(
            "Invalid predict_proba() output shape."
        )

    positive_index = classes.index(positive_class)
    positive_probabilities = probabilities[:, positive_index]

    if len(positive_probabilities) == 0:
        raise ModelEvaluationError(
            "Prediction input contains no records."
        )

    if not np.all(np.isfinite(positive_probabilities)):
        raise ModelEvaluationError(
            "Non-finite prediction probabilities detected."
        )

    if np.any(
        (positive_probabilities < 0)
        | (positive_probabilities > 1)
    ):
        raise ModelEvaluationError(
            "Predicted probabilities must lie within [0, 1]."
        )

    return positive_probabilities


def calculate_classification_metrics(
    y_true,
    probabilities,
    threshold=0.50
):
    """
    Calculate comprehensive binary classification metrics.

    Parameters
    ----------
    y_true : array-like
        Actual binary outcomes.
    probabilities : array-like
        Positive-class purchase probabilities.
    threshold : float
        Classification decision threshold.

    Returns
    -------
    dict
        JSON-compatible evaluation metrics.
    """
    if not 0 < threshold < 1:
        raise ValueError(
            "Classification threshold must be between 0 and 1."
        )

    y = np.asarray(y_true)
    p = np.asarray(probabilities, dtype=float)

    if y.ndim != 1 or p.ndim != 1:
        raise ModelEvaluationError(
            "Targets and probabilities must be 1-dimensional."
        )

    if len(y) == 0 or len(y) != len(p):
        raise ModelEvaluationError(
            "Targets and probabilities must have equal "
            "nonzero lengths."
        )

    if pd.isna(y).any() or not set(np.unique(y)).issubset({0, 1}):
        raise ModelEvaluationError(
            "Actual targets must contain only non-null 0/1 labels."
        )

    if not np.all(np.isfinite(p)):
        raise ModelEvaluationError(
            "Prediction probabilities contain non-finite values."
        )

    if np.any((p < 0) | (p > 1)):
        raise ModelEvaluationError(
            "Prediction probabilities must be between 0 and 1."
        )

    predictions = (p >= threshold).astype(int)

    # ROC-AUC and average precision require both target classes.
    both_classes = len(np.unique(y)) == 2

    metrics = {
        "sample_count": int(len(y)),
        "positive_count": int(np.sum(y == 1)),
        "negative_count": int(np.sum(y == 0)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y, predictions)
        ),
        "precision": float(
            precision_score(y, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y, predictions, zero_division=0)
        ),
        "f1": float(
            f1_score(y, predictions, zero_division=0)
        ),
        "roc_auc": (
            float(roc_auc_score(y, p))
            if both_classes else None
        ),
        "average_precision": (
            float(average_precision_score(y, p))
            if both_classes else None
        ),
        "log_loss": float(
            log_loss(y, np.column_stack([1 - p, p]), labels=[0, 1])
        ),
        "brier_score": float(
            brier_score_loss(y, p)
        ),
        "confusion_matrix": confusion_matrix(
            y, predictions, labels=[0, 1]
        ).tolist(),
        "classification_report": classification_report(
            y,
            predictions,
            labels=[0, 1],
            target_names=["Not Purchased", "Purchased"],
            output_dict=True,
            zero_division=0
        )
    }

    return metrics


def evaluate_classifier(
    model,
    X,
    y_true,
    threshold=0.50
):
    """Evaluate a fitted probabilistic classifier."""
    probabilities = get_positive_probabilities(model, X)

    metrics = calculate_classification_metrics(
        y_true,
        probabilities,
        threshold
    )

    get_logger().info(
        "Model evaluated: samples=%d, F1=%.4f, AP=%s",
        metrics["sample_count"],
        metrics["f1"],
        metrics["average_precision"]
    )

    return metrics


def evaluate_thresholds(
    y_true,
    probabilities,
    thresholds=None
):
    """
    Compare classification thresholds using validation data.

    Never use final test labels to choose a threshold.
    """
    if thresholds is None:
        thresholds = np.round(
            np.arange(0.10, 0.91, 0.05),
            2
        )

    records = []

    for threshold in thresholds:
        metrics = calculate_classification_metrics(
            y_true,
            probabilities,
            threshold=float(threshold)
        )

        records.append({
            "threshold": metrics["threshold"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"]
        })

    return pd.DataFrame(records)


def optimize_decision_threshold(
    y_validation,
    validation_probabilities,
    metric="f1",
    thresholds=None
):
    """
    Select the best threshold on validation predictions.

    Supported objectives:
        f1, precision, recall, balanced_accuracy

    Ties are resolved by choosing the threshold closest to 0.5,
    then the lower threshold.
    """
    supported = {
        "f1",
        "precision",
        "recall",
        "balanced_accuracy"
    }

    if metric not in supported:
        raise ValueError(
            f"Unsupported threshold metric: {metric}. "
            f"Choose from {sorted(supported)}."
        )

    results = evaluate_thresholds(
        y_validation,
        validation_probabilities,
        thresholds
    )

    results = results.copy()
    results["distance_from_default"] = (
        results["threshold"] - 0.50
    ).abs()

    results = results.sort_values(
        by=[
            metric,
            "distance_from_default",
            "threshold"
        ],
        ascending=[False, True, True]
    ).reset_index(drop=True)

    best = results.iloc[0]

    return {
        "best_threshold": float(best["threshold"]),
        "optimization_metric": metric,
        "best_validation_score": float(best[metric]),
        "threshold_comparison": results.drop(
            columns=["distance_from_default"]
        )
    }


def predict_customer_purchase(
    model,
    customer_data,
    threshold=0.50
):
    """
    Generate customer purchase predictions.

    Accepts a dictionary representing one customer or
    a DataFrame representing multiple customers.
    """
    if not 0 < threshold < 1:
        raise ValueError(
            "Prediction threshold must be between 0 and 1."
        )

    if isinstance(customer_data, dict):
        customer_data = pd.DataFrame([customer_data])

    if not isinstance(customer_data, pd.DataFrame):
        raise TypeError(
            "customer_data must be a dictionary or DataFrame."
        )

    if customer_data.empty:
        raise ValueError(
            "Customer input cannot be empty."
        )

    if customer_data.columns.duplicated().any():
        raise ValueError(
            "Duplicate customer input columns are not allowed."
        )

    probabilities = get_positive_probabilities(
        model,
        customer_data
    )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    result = pd.DataFrame(
        {
            "purchase_probability": probabilities,
            "predicted_class": predictions,
            "prediction_label": np.where(
                predictions == 1,
                "Likely to Purchase",
                "Unlikely to Purchase"
            )
        },
        index=customer_data.index
    )

    return result


def compare_model_results(model_results):
    """
    Compare model evaluation summaries.

    Parameters
    ----------
    model_results : dict
        Mapping of model names to evaluation dictionaries.

    Returns
    -------
    pandas.DataFrame
        Model comparison sorted by average precision.
    """
    if not isinstance(model_results, dict) or not model_results:
        raise ValueError(
            "model_results must be a nonempty dictionary."
        )

    rows = []

    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
        "brier_score",
        "log_loss"
    ]

    for model_name, metrics in model_results.items():
        if not isinstance(metrics, dict):
            raise TypeError(
                f"Metrics for '{model_name}' must be a dictionary."
            )

        row = {"model": model_name}

        for metric_name in metric_names:
            row[metric_name] = metrics.get(metric_name)

        rows.append(row)

    comparison = pd.DataFrame(rows)

    return comparison.sort_values(
        by="average_precision",
        ascending=False,
        na_position="last"
    ).reset_index(drop=True)



# ================================================================
# CENTRALIZED UTILITY MODULE
# SECTION 6 — MODEL ARTIFACTS & EXPERIMENT TRACKING
# ================================================================


def save_model(model, path):
    """
    Save a fitted model pipeline and generate its SHA-256 checksum.

    The artifact should include both preprocessing and
    classification steps to ensure inference consistency.

    Returns
    -------
    dict
        Artifact path, size and checksum.
    """
    destination = Path(path).expanduser()
    ensure_directory(destination.parent)

    temporary_path = None

    try:
        # Write to a temporary file before replacing the target.
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=str(destination.parent),
            prefix=".model_tmp_",
            suffix=".joblib",
            delete=False
        ) as handle:
            temporary_path = Path(handle.name)

        joblib.dump(model, temporary_path)

        if temporary_path.stat().st_size == 0:
            raise ArtifactError(
                "Serialized model artifact is empty."
            )

        os.replace(temporary_path, destination)

    except Exception as exc:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass

        raise ArtifactError(
            f"Unable to save model '{destination}': {exc}"
        ) from exc

    metadata = {
        "path": str(destination),
        "size_bytes": int(destination.stat().st_size),
        "sha256": sha256_file(destination),
        "saved_at_utc": datetime.now(
            timezone.utc
        ).isoformat()
    }

    get_logger().info(
        "Model artifact saved: %s",
        destination
    )

    return metadata


def validate_model_artifact(
    path,
    expected_sha256=None
):
    """
    Validate a serialized model artifact without loading it.

    Verifies file existence, nonzero size and optional checksum.

    A checksum verifies integrity, not trustworthiness.
    """
    file_path = Path(path).expanduser()

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Model artifact not found: {file_path}"
        )

    size_bytes = file_path.stat().st_size

    if size_bytes == 0:
        raise ArtifactError(
            f"Model artifact is empty: {file_path}"
        )

    actual_sha256 = sha256_file(file_path)

    if expected_sha256 is not None:
        if actual_sha256.lower() != expected_sha256.lower():
            raise ArtifactError(
                "Model artifact checksum mismatch. "
                f"File: {file_path}"
            )

    return {
        "valid": True,
        "path": str(file_path),
        "size_bytes": int(size_bytes),
        "sha256": actual_sha256
    }


def load_model(
    path,
    expected_sha256=None,
    trusted_source=False
):
    """
    Load a trusted model artifact after integrity validation.

    SECURITY WARNING:
        Joblib uses pickle internally. Loading untrusted files
        can execute arbitrary Python code.

    Set trusted_source=True only for artifacts produced by
    the project's controlled training pipeline.
    """
    if not trusted_source:
        raise ArtifactError(
            "Model loading blocked: trusted_source=False. "
            "Only load artifacts from the controlled "
            "project training pipeline."
        )

    validation = validate_model_artifact(
        path,
        expected_sha256=expected_sha256
    )

    try:
        model = joblib.load(validation["path"])

    except Exception as exc:
        raise ArtifactError(
            "Unable to deserialize trusted model artifact "
            f"'{path}': {exc}"
        ) from exc

    if not hasattr(model, "predict"):
        raise ArtifactError(
            "Loaded artifact does not implement predict()."
        )

    get_logger().info(
        "Trusted model artifact loaded: %s",
        path
    )

    return model


def save_experiment_results(
    results,
    path,
    experiment_id=None
):
    """
    Append an experiment record to a CSV tracking file.

    Existing experiment records are preserved.

    Parameters
    ----------
    results : dict
        Experiment metadata, parameters and metrics.
    path : str or Path
        Destination CSV file.
    experiment_id : str or None
        Optional unique experiment identifier.
    """
    if not isinstance(results, dict) or not results:
        raise ValueError(
            "Experiment results must be a nonempty dictionary."
        )

    destination = Path(path).expanduser()
    ensure_directory(destination.parent)

    record = results.copy()

    record["experiment_id"] = (
        experiment_id
        if experiment_id is not None
        else (
            "exp_"
            + datetime.now(timezone.utc).strftime(
                "%Y%m%d_%H%M%S_%f"
            )
        )
    )

    record["recorded_at_utc"] = (
        datetime.now(timezone.utc).isoformat()
    )

    # Serialize nested parameter dictionaries and metric objects
    # so that the CSV remains machine-readable.
    for key, value in record.items():
        if isinstance(value, (dict, list, tuple)):
            record[key] = json.dumps(
                value,
                ensure_ascii=False,
                default=str
            )

    try:
        new_record = pd.DataFrame([record])

        if destination.exists():
            existing = pd.read_csv(destination)

            if "experiment_id" in existing.columns:
                if record["experiment_id"] in set(
                    existing["experiment_id"].astype(str)
                ):
                    raise ArtifactError(
                        "Duplicate experiment ID detected: "
                        f"{record['experiment_id']}"
                    )

            combined = pd.concat(
                [existing, new_record],
                ignore_index=True
            )
        else:
            combined = new_record

        # Write to a temporary file before replacing the CSV.
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(destination.parent),
            prefix=".experiment_tmp_",
            suffix=".csv",
            delete=False
        ) as handle:
            temporary_path = Path(handle.name)

        try:
            combined.to_csv(
                temporary_path,
                index=False,
                encoding="utf-8"
            )

            os.replace(
                temporary_path,
                destination
            )

        finally:
            if temporary_path.exists():
                temporary_path.unlink(missing_ok=True)

    except (OSError, ValueError, TypeError) as exc:
        raise ArtifactError(
            f"Unable to save experiment results: {exc}"
        ) from exc

    get_logger().info(
        "Experiment recorded: %s",
        record["experiment_id"]
    )

    return {
        "experiment_id": record["experiment_id"],
        "path": str(destination),
        "total_experiments": int(len(combined))
    }


def load_experiment_results(path):
    """Load previously recorded experiment results."""
    file_path = Path(path).expanduser()

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Experiment tracking file not found: {file_path}"
        )

    try:
        experiments = pd.read_csv(file_path)

    except (OSError, pd.errors.ParserError) as exc:
        raise ArtifactError(
            f"Unable to load experiment results: {exc}"
        ) from exc

    return experiments


def generate_pipeline_report(
    project_name,
    model_name,
    metrics,
    artifact_metadata=None,
    output_path=None
):
    """
    Generate a JSON-compatible model evaluation report.

    Metrics should be computed on the appropriate validation
    or holdout dataset and labeled accordingly.
    """
    if not isinstance(metrics, dict):
        raise TypeError(
            "metrics must be a dictionary."
        )

    report = {
        "project": project_name,
        "model": model_name,
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "metrics": metrics,
        "artifact": artifact_metadata
    }

    if output_path is not None:
        save_json(report, output_path)

    return report


def update_artifact_manifest(
    artifact_name,
    artifact_metadata,
    manifest_path
):
    """
    Register an artifact in a centralized JSON manifest.

    Existing artifact records are preserved.
    """
    if not artifact_name:
        raise ValueError(
            "artifact_name cannot be empty."
        )

    if not isinstance(artifact_metadata, dict):
        raise TypeError(
            "artifact_metadata must be a dictionary."
        )

    manifest_file = Path(manifest_path).expanduser()

    if manifest_file.exists():
        manifest = load_json(manifest_file)
    else:
        manifest = {
            "project": "Visit with Us — Enterprise MLOps",
            "artifacts": {}
        }

    manifest.setdefault("artifacts", {})

    manifest["artifacts"][artifact_name] = {
        **artifact_metadata,
        "registered_at_utc": datetime.now(
            timezone.utc
        ).isoformat()
    }

    manifest["updated_at_utc"] = (
        datetime.now(timezone.utc).isoformat()
    )

    save_json(
        manifest,
        manifest_file
    )

    return manifest



# ============================================================
# SECTION 7A — CENTRALIZED EDA VISUALIZATION UTILITIES
# ============================================================

# Consistent project-wide colors.
CHART_COLORS = {
    "blue": "#2563EB",
    "teal": "#14B8A6",
    "purple": "#8B5CF6",
    "amber": "#F59E0B",
    "coral": "#EF4444",
    "green": "#10B981",
    "dark": "#1E293B"
}

TARGET_COLORS = {
    0: CHART_COLORS["blue"],
    1: CHART_COLORS["green"]
}

TARGET_LABELS = {
    0: "Not Purchased",
    1: "Purchased"
}


def save_figure(fig, save_path=None, dpi=300):
    """
    Export a Matplotlib figure at publication quality.

    Returns the saved path or None if saving is disabled.
    """
    if save_path is None:
        return None

    destination = Path(save_path)
    ensure_directory(destination.parent)

    try:
        fig.savefig(
            destination,
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white"
        )
    except OSError as exc:
        raise ArtifactError(
            f"Unable to save chart '{destination}': {exc}"
        ) from exc

    get_logger().info("Chart saved: %s", destination)
    return destination


def plot_target_distribution(
    dataframe,
    target_column="ProdTaken",
    save_path=None
):
    """Plot purchase counts and percentages with a legend."""
    if target_column not in dataframe.columns:
        raise DatasetValidationError(
            f"Target column missing: {target_column}"
        )

    counts = dataframe[target_column].value_counts().reindex(
        [0, 1], fill_value=0
    )

    total = int(counts.sum())

    if total == 0:
        raise DatasetValidationError(
            "Cannot plot an empty target distribution."
        )

    fig, ax = plt.subplots(figsize=(10, 6))

    labels = [TARGET_LABELS[i] for i in [0, 1]]
    colors = [TARGET_COLORS[i] for i in [0, 1]]

    bars = ax.bar(
        labels,
        counts.values,
        color=colors,
        width=0.55,
        edgecolor="white",
        linewidth=1.5
    )

    for bar, count in zip(bars, counts.values):
        ax.annotate(
            f"{count:,}\n({count / total:.1%})",
            xy=(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height()
            ),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11
        )

    ax.set_title(
        "Customer Purchase Distribution",
        fontsize=16,
        fontweight="bold",
        pad=18
    )
    ax.set_xlabel("Purchase Outcome", fontsize=12)
    ax.set_ylabel("Number of Customers", fontsize=12)

    ax.set_ylim(0, max(counts.max() * 1.20, 1))
    ax.grid(axis="y", linestyle="--", alpha=0.20)
    ax.set_axisbelow(True)

    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color=colors[i],
                          label=labels[i])
            for i in range(2)
        ],
        title="Purchase Status",
        loc="upper right"
    )

    fig.tight_layout()
    save_figure(fig, save_path)

    return fig, ax


def plot_numeric_distribution(
    dataframe,
    column,
    target_column="ProdTaken",
    bins=30,
    save_path=None
):
    """
    Plot a numerical feature by purchase outcome.

    Histograms use density normalization to support
    comparison between imbalanced target classes.
    """
    for required_column in [column, target_column]:
        if required_column not in dataframe.columns:
            raise DatasetValidationError(
                f"Missing column: {required_column}"
            )

    numeric_values = pd.to_numeric(
        dataframe[column],
        errors="coerce"
    )

    valid = dataframe.copy()
    valid[column] = numeric_values
    valid = valid.dropna(subset=[column, target_column])

    valid = valid.loc[
        np.isfinite(valid[column].to_numpy())
    ]

    if valid.empty:
        raise DatasetValidationError(
            f"No valid numeric values for '{column}'."
        )

    fig, ax = plt.subplots(figsize=(11, 6))

    for target_value in [0, 1]:
        values = valid.loc[
            valid[target_column] == target_value, column
        ]

        if values.empty:
            continue

        ax.hist(
            values,
            bins=bins,
            density=True,
            alpha=0.55,
            color=TARGET_COLORS[target_value],
            edgecolor="white",
            label=(
                f"{TARGET_LABELS[target_value]} "
                f"(n={len(values):,})"
            )
        )

    ax.set_title(
        f"{column} Distribution by Purchase Outcome",
        fontsize=16,
        fontweight="bold",
        pad=16
    )
    ax.set_xlabel(column, fontsize=12)
    ax.set_ylabel("Probability Density", fontsize=12)
    ax.legend(title="Purchase Status")
    ax.grid(axis="y", alpha=0.20, linestyle="--")
    ax.set_axisbelow(True)

    fig.tight_layout()
    save_figure(fig, save_path)

    return fig, ax


def plot_categorical_purchase_rate(
    dataframe,
    column,
    target_column="ProdTaken",
    save_path=None
):
    """Plot purchase conversion rates across categories."""
    for required_column in [column, target_column]:
        if required_column not in dataframe.columns:
            raise DatasetValidationError(
                f"Missing column: {required_column}"
            )

    valid = dataframe[[column, target_column]].copy()
    valid[column] = valid[column].astype("string").fillna(
        "Missing"
    )

    if not valid[target_column].isin([0, 1]).all():
        raise DatasetValidationError(
            "Target must contain only binary 0/1 values."
        )

    summary = (
        valid.groupby(column, dropna=False)[target_column]
        .agg(["mean", "count"])
        .sort_values("mean", ascending=False)
    )

    if summary.empty:
        raise DatasetValidationError(
            f"No categories available for '{column}'."
        )

    fig, ax = plt.subplots(
        figsize=(11, max(5, len(summary) * 0.55))
    )

    palette = [
        CHART_COLORS["teal"]
        if rate >= valid[target_column].mean()
        else CHART_COLORS["purple"]
        for rate in summary["mean"]
    ]

    bars = ax.barh(
        summary.index.astype(str),
        summary["mean"] * 100,
        color=palette,
        edgecolor="white"
    )

    for bar, rate, count in zip(
        bars,
        summary["mean"],
        summary["count"]
    ):
        ax.text(
            bar.get_width() + 0.6,
            bar.get_y() + bar.get_height() / 2,
            f"{rate:.1%} (n={count:,})",
            va="center",
            fontsize=9
        )

    overall_rate = valid[target_column].mean() * 100

    ax.axvline(
        overall_rate,
        color=CHART_COLORS["coral"],
        linestyle="--",
        linewidth=2,
        label=f"Overall Purchase Rate ({overall_rate:.1f}%)"
    )

    ax.set_xlim(
        0,
        max(105, summary["mean"].max() * 100 + 20)
    )
    ax.invert_yaxis()

    ax.set_title(
        f"Purchase Rate by {column}",
        fontsize=16,
        fontweight="bold",
        pad=16
    )
    ax.set_xlabel("Purchase Rate (%)", fontsize=12)
    ax.set_ylabel(column, fontsize=12)
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.20, linestyle="--")
    ax.set_axisbelow(True)

    fig.tight_layout()
    save_figure(fig, save_path)

    return fig, ax



# ================================================================
# SECTION 7B — CORRELATION & EVALUATION VISUALIZATIONS
# ================================================================


def plot_correlation_heatmap(
    dataframe,
    method="spearman",
    save_path=None
):
    """
    Plot a correlation heatmap for numerical features.

    Spearman correlation is the default because it can
    capture monotonic relationships without requiring
    linearity or normally distributed variables.

    Correlation does not establish causation.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            "Correlation analysis requires a DataFrame."
        )

    numeric_data = dataframe.select_dtypes(
        include=["number"]
    )

    # Exclude identifiers when present.
    numeric_data = numeric_data.drop(
        columns=["CustomerID", "Unnamed: 0"],
        errors="ignore"
    )

    if numeric_data.shape[1] < 2:
        raise DatasetValidationError(
            "At least two numerical features are required "
            "for a correlation heatmap."
        )

    correlation = numeric_data.corr(method=method)

    if correlation.isna().all().all():
        raise DatasetValidationError(
            "Correlation matrix contains no valid values."
        )

    fig, ax = plt.subplots(
        figsize=(
            max(10, correlation.shape[1] * 0.75),
            max(8, correlation.shape[0] * 0.65)
        )
    )

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.6,
        linecolor="white",
        square=True,
        cbar_kws={
            "label": f"{method.title()} Correlation"
        },
        annot_kws={"fontsize": 8},
        ax=ax
    )

    ax.set_title(
        f"Numerical Feature Correlation — {method.title()}",
        fontsize=16,
        fontweight="bold",
        pad=20
    )

    ax.set_xlabel("Numerical Features", fontsize=12)
    ax.set_ylabel("Numerical Features", fontsize=12)

    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha="right"
    )

    plt.setp(
        ax.get_yticklabels(),
        rotation=0
    )

    fig.tight_layout()
    save_figure(fig, save_path)

    return fig, ax


def plot_confusion_matrix_chart(
    y_true,
    y_pred,
    title="Customer Purchase — Confusion Matrix",
    save_path=None
):
    """
    Plot a labeled confusion matrix with counts.

    Rows represent actual classes.
    Columns represent predicted classes.
    """
    from sklearn.metrics import confusion_matrix

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    fig, ax = plt.subplots(figsize=(9, 7))

    sns.heatmap(
        matrix,
        annot=True,
        fmt=",d",
        cmap="Blues",
        linewidths=1.5,
        linecolor="white",
        xticklabels=[
            "Not Purchased",
            "Purchased"
        ],
        yticklabels=[
            "Not Purchased",
            "Purchased"
        ],
        cbar_kws={
            "label": "Number of Customers"
        },
        annot_kws={
            "fontsize": 14,
            "fontweight": "bold"
        },
        ax=ax
    )

    ax.set_title(
        title,
        fontsize=16,
        fontweight="bold",
        pad=18
    )

    ax.set_xlabel(
        "Predicted Purchase Outcome",
        fontsize=12
    )

    ax.set_ylabel(
        "Actual Purchase Outcome",
        fontsize=12
    )

    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)

    fig.tight_layout()
    save_figure(fig, save_path)

    return fig, ax


def plot_roc_curve_chart(
    y_true,
    probabilities,
    model_name="Classification Model",
    save_path=None
):
    """
    Plot the ROC curve with AUC and random baseline.

    Requires both positive and negative target classes.
    """
    from sklearn.metrics import (
        roc_curve,
        roc_auc_score
    )

    y = np.asarray(y_true)
    p = np.asarray(probabilities)

    if len(np.unique(y)) != 2:
        raise ModelEvaluationError(
            "ROC curve requires both target classes."
        )

    if len(y) != len(p):
        raise ModelEvaluationError(
            "Target and probability lengths do not match."
        )

    fpr, tpr, _ = roc_curve(y, p)
    auc_score = roc_auc_score(y, p)

    fig, ax = plt.subplots(figsize=(9, 7))

    ax.plot(
        fpr,
        tpr,
        color=CHART_COLORS["blue"],
        linewidth=3,
        label=f"{model_name} (ROC-AUC = {auc_score:.3f})"
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color=CHART_COLORS["coral"],
        linewidth=2,
        label="Random Classifier (AUC = 0.500)"
    )

    ax.fill_between(
        fpr,
        tpr,
        alpha=0.12,
        color=CHART_COLORS["blue"]
    )

    ax.set_title(
        f"ROC Curve — {model_name}",
        fontsize=16,
        fontweight="bold",
        pad=18
    )

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=12)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)

    ax.grid(
        linestyle="--",
        alpha=0.20
    )

    ax.legend(
        loc="lower right",
        title="Model Performance",
        frameon=True
    )

    fig.tight_layout()
    save_figure(fig, save_path)

    return fig, ax


def plot_precision_recall_curve_chart(
    y_true,
    probabilities,
    model_name="Classification Model",
    save_path=None
):
    """
    Plot the precision-recall curve and positive-class baseline.

    Average Precision is displayed as the summary score.
    """
    from sklearn.metrics import (
        precision_recall_curve,
        average_precision_score
    )

    y = np.asarray(y_true)
    p = np.asarray(probabilities)

    if len(np.unique(y)) != 2:
        raise ModelEvaluationError(
            "Precision-recall curve requires both classes."
        )

    if len(y) != len(p):
        raise ModelEvaluationError(
            "Target and probability lengths do not match."
        )

    precision, recall, _ = precision_recall_curve(y, p)

    ap_score = average_precision_score(y, p)
    baseline = float(np.mean(y == 1))

    fig, ax = plt.subplots(figsize=(9, 7))

    ax.plot(
        recall,
        precision,
        color=CHART_COLORS["green"],
        linewidth=3,
        label=f"{model_name} (AP = {ap_score:.3f})"
    )

    ax.axhline(
        baseline,
        color=CHART_COLORS["coral"],
        linestyle="--",
        linewidth=2,
        label=f"Class Prevalence ({baseline:.1%})"
    )

    ax.fill_between(
        recall,
        precision,
        alpha=0.12,
        color=CHART_COLORS["green"]
    )

    ax.set_title(
        f"Precision-Recall Curve — {model_name}",
        fontsize=16,
        fontweight="bold",
        pad=18
    )

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)

    ax.grid(
        linestyle="--",
        alpha=0.20
    )

    ax.legend(
        loc="upper right",
        title="Model Performance",
        frameon=True
    )

    fig.tight_layout()
    save_figure(fig, save_path)

    return fig, ax



# ================================================================
# SECTION 7C — ADVANCED MODEL VISUALIZATION UTILITIES
# ================================================================


def plot_feature_importance(
    feature_names,
    importance_values,
    top_n=15,
    title="Top Predictive Features",
    save_path=None
):
    """
    Visualize ranked feature importance.

    Parameters
    ----------
    feature_names : sequence
        Names of features corresponding to importance values.
    importance_values : sequence
        Feature importance scores.
    top_n : int
        Maximum number of features displayed.
    title : str
        Descriptive chart title.
    save_path : str or Path, optional
        Output image location.

    Notes
    -----
    Importance values must come from a documented method
    such as permutation importance or model-native importance.

    Feature importance describes predictive association,
    not causal influence.
    """
    names = list(feature_names)
    values = np.asarray(importance_values, dtype=float)

    if not names or len(names) != len(values):
        raise ValueError(
            "Feature names and importance values must "
            "have equal, nonzero lengths."
        )

    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError(
            "Feature importance values must be finite "
            "and one-dimensional."
        )

    if not isinstance(top_n, int) or top_n < 1:
        raise ValueError("top_n must be a positive integer.")

    importance_df = pd.DataFrame({
        "feature": names,
        "importance": values
    })

    importance_df["absolute_importance"] = (
        importance_df["importance"].abs()
    )

    importance_df = (
        importance_df
        .sort_values("absolute_importance", ascending=False)
        .head(top_n)
        .sort_values("absolute_importance", ascending=True)
    )

    fig, ax = plt.subplots(
        figsize=(11, max(6, len(importance_df) * 0.45))
    )

    bar_colors = [
        CHART_COLORS["blue"]
        if value >= 0
        else CHART_COLORS["coral"]
        for value in importance_df["importance"]
    ]

    bars = ax.barh(
        importance_df["feature"],
        importance_df["importance"],
        color=bar_colors,
        edgecolor="white",
        linewidth=0.8
    )

    for bar, value in zip(
        bars,
        importance_df["importance"]
    ):
        offset = max(
            float(importance_df["absolute_importance"].max()) * 0.02,
            0.001
        )

        ax.text(
            bar.get_width() + (
                offset if value >= 0 else -offset
            ),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.4f}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=9
        )

    ax.axvline(
        0,
        color=CHART_COLORS["dark"],
        linewidth=1
    )

    ax.set_title(
        title,
        fontsize=16,
        fontweight="bold",
        pad=18
    )

    ax.set_xlabel(
        "Feature Importance Score",
        fontsize=12
    )
    ax.set_ylabel("Predictive Features", fontsize=12)

    ax.grid(axis="x", linestyle="--", alpha=0.20)
    ax.set_axisbelow(True)

    ax.legend(
        handles=[
            plt.Rectangle(
                (0, 0), 1, 1,
                color=CHART_COLORS["blue"],
                label="Positive Importance"
            ),
            plt.Rectangle(
                (0, 0), 1, 1,
                color=CHART_COLORS["coral"],
                label="Negative Importance"
            )
        ],
        loc="lower right"
    )

    fig.tight_layout()
    save_figure(fig, save_path)

    return fig, ax


def plot_model_comparison(
    comparison_df,
    metrics=None,
    title="Machine Learning Model Comparison",
    save_path=None
):
    """
    Compare multiple models using grouped metric bars.

    The input must contain a 'model' column and the
    requested evaluation metric columns.
    """
    if not isinstance(comparison_df, pd.DataFrame):
        raise TypeError(
            "comparison_df must be a pandas DataFrame."
        )

    if comparison_df.empty:
        raise ValueError(
            "Model comparison data cannot be empty."
        )

    if metrics is None:
        metrics = [
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "average_precision"
        ]

    required = ["model", *metrics]

    missing = [
        column
        for column in required
        if column not in comparison_df.columns
    ]

    if missing:
        raise ValueError(
            f"Model comparison columns missing: {missing}"
        )

    if comparison_df["model"].duplicated().any():
        raise ValueError(
            "Model comparison requires unique model labels."
        )

    plot_df = comparison_df[required].copy()

    for metric in metrics:
        plot_df[metric] = pd.to_numeric(
            plot_df[metric],
            errors="raise"
        )

    values = plot_df[metrics].to_numpy(dtype=float)

    if not np.all(np.isfinite(values)):
        raise ValueError(
            "Model comparison contains non-finite metric values."
        )

    if np.any((values < 0) | (values > 1)):
        raise ValueError(
            "Comparison metrics must be normalized to [0, 1]."
        )

    x = np.arange(len(plot_df))
    width = 0.80 / len(metrics)

    palette = [
        CHART_COLORS["blue"],
        CHART_COLORS["teal"],
        CHART_COLORS["purple"],
        CHART_COLORS["amber"],
        CHART_COLORS["green"],
        CHART_COLORS["coral"]
    ]

    fig, ax = plt.subplots(
        figsize=(max(12, len(plot_df) * 1.7), 7)
    )

    for index, metric in enumerate(metrics):
        offset = (
            index - (len(metrics) - 1) / 2
        ) * width

        bars = ax.bar(
            x + offset,
            plot_df[metric],
            width=width,
            color=palette[index % len(palette)],
            label=metric.replace("_", " ").title(),
            edgecolor="white"
        )

        for bar in bars:
            height = bar.get_height()

            ax.annotate(
                f"{height:.2f}",
                (
                    bar.get_x() + bar.get_width() / 2,
                    height
                ),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90
            )

    ax.set_xticks(x)
    ax.set_xticklabels(
        plot_df["model"].astype(str),
        rotation=20,
        ha="right"
    )

    ax.set_ylim(0, 1.15)

    ax.set_title(
        title,
        fontsize=16,
        fontweight="bold",
        pad=18
    )

    ax.set_xlabel("Machine Learning Models", fontsize=12)
    ax.set_ylabel("Evaluation Metric Score", fontsize=12)

    ax.legend(
        title="Evaluation Metrics",
        loc="upper left",
        bbox_to_anchor=(1.01, 1)
    )

    ax.grid(axis="y", linestyle="--", alpha=0.20)
    ax.set_axisbelow(True)

    fig.tight_layout()
    save_figure(fig, save_path)

    return fig, ax


def plot_threshold_analysis(
    threshold_df,
    title="Classification Threshold Optimization",
    save_path=None
):
    """
    Visualize precision, recall and F1 across thresholds.

    Threshold selection must be based on validation data,
    not the final holdout test dataset.
    """
    required = [
        "threshold",
        "precision",
        "recall",
        "f1"
    ]

    if not isinstance(threshold_df, pd.DataFrame):
        raise TypeError(
            "threshold_df must be a DataFrame."
        )

    if threshold_df.empty:
        raise ValueError(
            "Threshold analysis data cannot be empty."
        )

    missing = [
        column
        for column in required
        if column not in threshold_df.columns
    ]

    if missing:
        raise ValueError(
            f"Threshold analysis columns missing: {missing}"
        )

    plot_df = threshold_df[required].copy()

    for column in required:
        plot_df[column] = pd.to_numeric(
            plot_df[column],
            errors="raise"
        )

    if not np.all(
        np.isfinite(plot_df.to_numpy(dtype=float))
    ):
        raise ValueError(
            "Threshold analysis contains invalid numeric values."
        )

    if np.any(
        (plot_df.to_numpy(dtype=float) < 0)
        | (plot_df.to_numpy(dtype=float) > 1)
    ):
        raise ValueError(
            "Thresholds and metric values must be in [0, 1]."
        )

    plot_df = plot_df.sort_values("threshold")

    fig, ax = plt.subplots(figsize=(11, 7))

    series = [
        ("precision", CHART_COLORS["blue"]),
        ("recall", CHART_COLORS["teal"]),
        ("f1", CHART_COLORS["purple"])
    ]

    for metric, color in series:
        ax.plot(
            plot_df["threshold"],
            plot_df[metric],
            color=color,
            linewidth=2.5,
            marker="o",
            markersize=5,
            label=metric.title()
        )

    best_index = plot_df["f1"].idxmax()
    best_threshold = float(
        plot_df.loc[best_index, "threshold"]
    )

    ax.axvline(
        best_threshold,
        color=CHART_COLORS["coral"],
        linestyle="--",
        linewidth=2,
        label=f"Best F1 Threshold ({best_threshold:.2f})"
    )

    ax.set_title(
        title,
        fontsize=16,
        fontweight="bold",
        pad=18
    )

    ax.set_xlabel("Classification Threshold", fontsize=12)
    ax.set_ylabel("Evaluation Metric Score", fontsize=12)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)

    ax.legend(
        title="Validation Performance",
        loc="best"
    )

    ax.grid(
        linestyle="--",
        alpha=0.20
    )
    ax.set_axisbelow(True)

    fig.tight_layout()
    save_figure(fig, save_path)

    return fig, ax
