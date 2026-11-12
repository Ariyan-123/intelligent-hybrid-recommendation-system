
"""
Intelligent Hybrid Recommendation System
Performance Analysis and Evaluation
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)


# ============================================================
# Configuration
# ============================================================

REQUIRED_FEEDBACK_COLUMNS = {
    "user_id",
    "matched_user_id",
    "action",
    "timestamp",
}

STAGE_NAMES = [
    "Early",
    "Middle-Early",
    "Middle-Late",
    "Late",
]


# ============================================================
# Common Validation
# ============================================================

def validate_scores(scores, name="Scores"):
    """Validate that scores are numeric and between 0 and 1."""

    try:
        values = np.asarray(scores, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{name} must contain numeric values."
        ) from error

    if not np.isfinite(values).all():
        raise ValueError(
            f"{name} contain missing or invalid values."
        )

    if np.any((values < 0) | (values > 1)):
        raise ValueError(
            f"{name} must be between 0 and 1."
        )

    return values


# ============================================================
# Load Feedback Data
# ============================================================

def load_feedback(project_root):
    """Load and chronologically sort feedback records."""

    project_root = Path(project_root)

    feedback_path = (
        project_root / "data" / "feedback.csv"
    )

    if not feedback_path.is_file():
        raise FileNotFoundError(
            f"Feedback dataset not found: {feedback_path}"
        )

    try:
        feedback_df = pd.read_csv(feedback_path)

    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as error:
        raise RuntimeError(
            f"Unable to load feedback dataset: {error}"
        ) from error

    missing_columns = (
        REQUIRED_FEEDBACK_COLUMNS - set(feedback_df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing feedback columns: {sorted(missing_columns)}"
        )

    if feedback_df.empty:
        raise ValueError(
            "Feedback dataset contains no records."
        )

    feedback_df = feedback_df.copy()

    # Normalize IDs.
    for column in ["user_id", "matched_user_id"]:
        feedback_df[column] = (
            feedback_df[column]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.upper()
        )

        if feedback_df[column].eq("").any():
            raise ValueError(
                f"Blank values found in '{column}'."
            )

    # Validate feedback actions.
    feedback_df["action"] = pd.to_numeric(
        feedback_df["action"],
        errors="coerce",
    )

    if feedback_df["action"].isna().any():
        raise ValueError(
            "Missing or invalid feedback actions found."
        )

    if not feedback_df["action"].isin([0, 1]).all():
        raise ValueError(
            "Feedback actions must be 0 (reject) or 1 (accept)."
        )

    feedback_df["action"] = feedback_df["action"].astype(int)

    # Parse timestamps.
    feedback_df["timestamp"] = pd.to_datetime(
        feedback_df["timestamp"],
        errors="coerce",
    )

    if feedback_df["timestamp"].isna().any():
        raise ValueError(
            "Missing or invalid feedback timestamps found."
        )

    feedback_df = (
        feedback_df
        .sort_values(
            "timestamp",
            kind="stable",
        )
        .reset_index(drop=True)
    )

    return feedback_df


# ============================================================
# Baseline Prediction
# ============================================================

def baseline_prediction(
    text_score,
    mbti_score,
    location_score,
):
    """
    Calculate compatibility using equal weights.

    Each matching component must be between 0 and 1.
    """

    features = validate_scores(
        [
            text_score,
            mbti_score,
            location_score,
        ],
        "Matching components",
    )

    baseline_weights = np.array(
        [1 / 3, 1 / 3, 1 / 3],
        dtype=float,
    )

    return float(
        np.dot(
            baseline_weights,
            features,
        )
    )


# ============================================================
# Adaptive Prediction
# ============================================================

def adaptive_prediction(
    text_score,
    mbti_score,
    location_score,
    weights,
):
    """
    Calculate compatibility using personalized weights.
    """

    features = validate_scores(
        [
            text_score,
            mbti_score,
            location_score,
        ],
        "Matching components",
    )

    required_keys = {
        "text",
        "mbti",
        "location",
    }

    if not isinstance(weights, dict):
        raise ValueError(
            "Weights must be supplied as a dictionary."
        )

    missing_keys = required_keys - set(weights.keys())

    if missing_keys:
        raise ValueError(
            f"Missing weight keys: {sorted(missing_keys)}"
        )

    try:
        weight_vector = np.array(
            [
                weights["text"],
                weights["mbti"],
                weights["location"],
            ],
            dtype=float,
        )

    except (TypeError, ValueError) as error:
        raise ValueError(
            "Personalized weights must be numeric."
        ) from error

    if not np.isfinite(weight_vector).all():
        raise ValueError(
            "Personalized weights contain invalid values."
        )

    if np.any(weight_vector < 0):
        raise ValueError(
            "Personalized weights cannot be negative."
        )

    total = float(weight_vector.sum())

    if total <= 1e-12:
        raise ValueError(
            "Personalized weights cannot sum to zero."
        )

    # Normalize weights to ensure their sum is 1.
    weight_vector = weight_vector / total

    return float(
        np.dot(
            weight_vector,
            features,
        )
    )


# ============================================================
# Convert Compatibility Score to Binary Prediction
# ============================================================

def convert_to_prediction(
    probability,
    threshold=0.5,
):
    """Convert a compatibility score into a binary prediction."""

    probability = float(
        validate_scores(
            [probability],
            "Probability",
        )[0]
    )

    threshold = float(
        validate_scores(
            [threshold],
            "Threshold",
        )[0]
    )

    return int(probability >= threshold)


# ============================================================
# Evaluate Predictions
# ============================================================

def evaluate_predictions(
    actual,
    predicted,
):
    """
    Calculate precision, recall and F1-score.

    Inputs must contain binary labels:
    1 = accepted/positive
    0 = rejected/negative
    """

    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    if actual.ndim != 1 or predicted.ndim != 1:
        raise ValueError(
            "Actual and predicted labels must be one-dimensional."
        )

    if len(actual) == 0:
        raise ValueError(
            "Evaluation datasets cannot be empty."
        )

    if len(actual) != len(predicted):
        raise ValueError(
            "Actual and predicted arrays must have equal lengths."
        )

    if not np.isin(actual, [0, 1]).all():
        raise ValueError(
            "Actual labels must contain only 0 and 1."
        )

    if not np.isin(predicted, [0, 1]).all():
        raise ValueError(
            "Predicted labels must contain only 0 and 1."
        )

    actual = actual.astype(int)
    predicted = predicted.astype(int)

    precision = precision_score(
        actual,
        predicted,
        zero_division=0,
    )

    recall = recall_score(
        actual,
        predicted,
        zero_division=0,
    )

    f1 = f1_score(
        actual,
        predicted,
        zero_division=0,
    )

    return {
        "Precision": float(precision),
        "Recall": float(recall),
        "F1 Score": float(f1),
    }


# ============================================================
# Chronological Feedback Analysis
# ============================================================

def analyze_feedback_over_time(feedback_df):
    """
    Divide chronologically sorted feedback into up to
    four approximately equal stages.

    Supports datasets with fewer than four records.
    """

    required_columns = {
        "action",
        "timestamp",
    }

    missing_columns = (
        required_columns - set(feedback_df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing analysis columns: {sorted(missing_columns)}"
        )

    if feedback_df.empty:
        return pd.DataFrame(
            columns=[
                "stage",
                "interactions",
                "accepts",
                "acceptance_rate",
            ]
        )

    feedback_df = feedback_df.copy()

    feedback_df["action"] = pd.to_numeric(
        feedback_df["action"],
        errors="coerce",
    )

    if (
        feedback_df["action"].isna().any()
        or not feedback_df["action"].isin([0, 1]).all()
    ):
        raise ValueError(
            "Invalid feedback actions found."
        )

    feedback_df["timestamp"] = pd.to_datetime(
        feedback_df["timestamp"],
        errors="coerce",
    )

    if feedback_df["timestamp"].isna().any():
        raise ValueError(
            "Invalid feedback timestamps found."
        )

    feedback_df = (
        feedback_df
        .sort_values(
            "timestamp",
            kind="stable",
        )
        .reset_index(drop=True)
    )

    # Split into up to four chronological groups.
    number_of_stages = min(
        len(feedback_df),
        len(STAGE_NAMES),
    )

    index_groups = np.array_split(
        np.arange(len(feedback_df)),
        number_of_stages,
    )

    stage_frames = []

    for index, group in enumerate(index_groups):

        stage_name = STAGE_NAMES[index]

        stage_df = feedback_df.iloc[group]

        stage_frames.append({
            "stage": stage_name,
            "interactions": int(len(stage_df)),
            "accepts": int(stage_df["action"].sum()),
            "acceptance_rate": float(
                stage_df["action"].mean() * 100
            ),
        })

    return pd.DataFrame(stage_frames)


# ============================================================
# Main Performance Analysis
# ============================================================

def run_performance_analysis(project_root):
    """Analyze feedback patterns and save the results."""

    project_root = Path(project_root)

    feedback_df = load_feedback(
        project_root
    )

    print("\nRECOMMENDATION PERFORMANCE ANALYSIS")
    print("=" * 60)

    total_interactions = len(feedback_df)

    accepted = int(
        (feedback_df["action"] == 1).sum()
    )

    rejected = int(
        (feedback_df["action"] == 0).sum()
    )

    acceptance_rate = (
        feedback_df["action"].mean() * 100
    )

    print(f"Total feedback interactions: {total_interactions}")
    print(f"Accepted recommendations: {accepted}")
    print(f"Rejected recommendations: {rejected}")
    print(f"Overall acceptance rate: {acceptance_rate:.2f}%")

    # --------------------------------------------------------
    # Analyze chronological feedback
    # --------------------------------------------------------

    stage_results = analyze_feedback_over_time(
        feedback_df
    )

    print("\nFEEDBACK OVER TIME")
    print("=" * 60)

    print(
        stage_results.to_string(index=False)
    )

    # --------------------------------------------------------
    # Compare first and last chronological stages
    # --------------------------------------------------------

    if len(stage_results) >= 2:

        first_rate = float(
            stage_results.iloc[0]["acceptance_rate"]
        )

        last_rate = float(
            stage_results.iloc[-1]["acceptance_rate"]
        )

        change = last_rate - first_rate

        print("\nFEEDBACK TREND")
        print("=" * 60)

        print(
            f"First-stage acceptance rate: {first_rate:.2f}%"
        )

        print(
            f"Last-stage acceptance rate: {last_rate:.2f}%"
        )

        print(
            f"Change: {change:+.2f} percentage points"
        )

    # --------------------------------------------------------
    # Save chronological results
    # --------------------------------------------------------

    outputs_dir = project_root / "outputs"

    outputs_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        outputs_dir / "feedback_over_time.csv"
    )

    stage_results.to_csv(
        output_path,
        index=False,
    )

    print("\nOUTPUT SAVED")
    print("=" * 60)

    print(output_path)

    print(
        "\nNote: Acceptance-rate trends describe observed "
        "feedback behavior. They do not independently prove "
        "that recommendation quality improved."
    )

    return stage_results


# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    run_performance_analysis(
        PROJECT_ROOT
    )
