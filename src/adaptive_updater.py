
"""
Intelligent Hybrid Recommendation System
Adaptive Feedback and Personalized Weight Updater
"""

from pathlib import Path
import os

import joblib
import numpy as np


# ============================================================
# Configuration
# ============================================================

DEFAULT_WEIGHTS = {
    "text": 1 / 3,
    "mbti": 1 / 3,
    "location": 1 / 3,
}

WEIGHT_KEYS = ("text", "mbti", "location")

DEFAULT_LEARNING_RATE = 0.05


# ============================================================
# Project Paths
# ============================================================

def _paths(project_root):
    """Return model and data directories."""

    project_root = Path(project_root)

    models_dir = project_root / "models"
    data_dir = project_root / "data"

    return models_dir, data_dir


# ============================================================
# Input Validation
# ============================================================

def _validate_score(value, name):
    """Validate a matching component score between 0 and 1."""

    try:
        score = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{name} must be a numeric value."
        ) from error

    if not np.isfinite(score):
        raise ValueError(
            f"{name} must be finite."
        )

    if not 0.0 <= score <= 1.0:
        raise ValueError(
            f"{name} must be between 0 and 1. Received: {score}"
        )

    return score


def _normalize_user_id(user_id):
    """Normalize user IDs consistently."""

    if user_id is None:
        raise ValueError("User ID cannot be empty.")

    normalized = str(user_id).strip().upper()

    if not normalized:
        raise ValueError("User ID cannot be empty.")

    return normalized


# ============================================================
# Project Weights Onto Probability Simplex
# ============================================================

def project_to_simplex(weights):
    """
    Project weights onto a valid probability simplex.

    Guarantees:
    - Exactly three weights.
    - No negative weights.
    - All weights sum to 1.
    """

    weights = np.asarray(weights, dtype=float)

    if weights.shape != (3,):
        raise ValueError(
            "Exactly three weights are required: "
            "text, mbti and location."
        )

    if not np.isfinite(weights).all():
        raise ValueError(
            "Weights contain invalid numeric values."
        )

    weights = np.maximum(weights, 0.0)

    total = float(weights.sum())

    if total <= 1e-12:
        return np.array(
            [1 / 3, 1 / 3, 1 / 3],
            dtype=float,
        )

    return weights / total


# ============================================================
# Validate Stored Weight Dictionary
# ============================================================

def _validate_weight_dictionary(weights):
    """Validate and normalize one user's stored weights."""

    if not isinstance(weights, dict):
        raise ValueError(
            "Stored personalized weights must be a dictionary."
        )

    missing_keys = set(WEIGHT_KEYS) - set(weights.keys())

    if missing_keys:
        raise ValueError(
            f"Missing weight keys: {sorted(missing_keys)}"
        )

    values = []

    for key in WEIGHT_KEYS:
        try:
            value = float(weights[key])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid stored weight for '{key}'."
            ) from error

        if not np.isfinite(value) or value < 0:
            raise ValueError(
                f"Stored weight '{key}' must be finite and non-negative."
            )

        values.append(value)

    normalized = project_to_simplex(values)

    return {
        key: float(value)
        for key, value in zip(WEIGHT_KEYS, normalized)
    }


# ============================================================
# Load Personalized Weights
# ============================================================

def load_personalized_weights(project_root):
    """
    Load personalized weights from the models directory.

    If the artifact does not exist, return an empty dictionary.
    Existing corrupted artifacts are not silently overwritten.
    """

    models_dir, _ = _paths(project_root)

    path = models_dir / "personalized_weights.joblib"

    if not path.is_file():
        return {}

    try:
        personalized_weights = joblib.load(path)

    except Exception as error:
        raise RuntimeError(
            f"Unable to load personalized weights from {path}: {error}"
        ) from error

    if not isinstance(personalized_weights, dict):
        raise ValueError(
            "Personalized weights artifact must contain a dictionary."
        )

    normalized_weights = {}

    for user_id, weights in personalized_weights.items():

        normalized_id = _normalize_user_id(user_id)

        if normalized_id in normalized_weights:
            raise ValueError(
                f"Duplicate user ID after normalization: {normalized_id}"
            )

        normalized_weights[normalized_id] = (
            _validate_weight_dictionary(weights)
        )

    return normalized_weights


# ============================================================
# Save Personalized Weights
# ============================================================

def save_personalized_weights(
    project_root,
    personalized_weights,
):
    """
    Save personalized weights safely.

    Uses a temporary file followed by atomic replacement
    to reduce the risk of a partially written artifact.
    """

    models_dir, _ = _paths(project_root)

    models_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = models_dir / "personalized_weights.joblib"

    if not isinstance(personalized_weights, dict):
        raise ValueError(
            "Personalized weights must be a dictionary."
        )

    validated_weights = {}

    for user_id, weights in personalized_weights.items():

        normalized_id = _normalize_user_id(user_id)

        if normalized_id in validated_weights:
            raise ValueError(
                f"Duplicate user ID: {normalized_id}"
            )

        validated_weights[normalized_id] = (
            _validate_weight_dictionary(weights)
        )

    temporary_path = path.with_name(
        path.name + ".tmp"
    )

    try:
        joblib.dump(
            validated_weights,
            temporary_path,
        )

        os.replace(
            temporary_path,
            path,
        )

    except Exception as error:
        if temporary_path.exists():
            temporary_path.unlink()

        raise RuntimeError(
            f"Unable to save personalized weights: {error}"
        ) from error


# ============================================================
# Update One User's Weights
# ============================================================

def update_user_weights(
    project_root,
    user_id,
    text_score,
    mbti_score,
    location_score,
    action,
    learning_rate=DEFAULT_LEARNING_RATE,
):
    """
    Update one user's personalized matching weights.

    Uses a feedback-driven gradient update.

    Parameters
    ----------
    project_root : str or Path
        Project root directory.

    user_id : str
        ID of the user providing feedback.

    text_score : float
        Text similarity score between 0 and 1.

    mbti_score : float
        Normalized MBTI compatibility score between 0 and 1.

    location_score : float
        Location compatibility score between 0 and 1.

    action : int
        1 = accepted recommendation.
        0 = rejected recommendation.

    learning_rate : float
        Positive learning rate, at most 1.

    Returns
    -------
    dict
        Updated personalized weights.
    """

    # --------------------------------------------------------
    # Validate inputs
    # --------------------------------------------------------

    user_id = _normalize_user_id(user_id)

    text_score = _validate_score(
        text_score,
        "Text score",
    )

    mbti_score = _validate_score(
        mbti_score,
        "MBTI score",
    )

    location_score = _validate_score(
        location_score,
        "Location score",
    )

    try:
        action = float(action)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Action must be either 0 or 1."
        ) from error

    if not np.isfinite(action) or action not in (0.0, 1.0):
        raise ValueError(
            "Action must be either 0 (reject) or 1 (accept)."
        )

    try:
        learning_rate = float(learning_rate)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Learning rate must be numeric."
        ) from error

    if (
        not np.isfinite(learning_rate)
        or not 0 < learning_rate <= 1
    ):
        raise ValueError(
            "Learning rate must be greater than 0 and at most 1."
        )

    # --------------------------------------------------------
    # Load current personalized weights
    # --------------------------------------------------------

    personalized_weights = load_personalized_weights(
        project_root
    )

    current = personalized_weights.get(
        user_id,
        DEFAULT_WEIGHTS.copy(),
    )

    weights = np.array(
        [current[key] for key in WEIGHT_KEYS],
        dtype=float,
    )

    # --------------------------------------------------------
    # Matching feature vector
    # --------------------------------------------------------

    features = np.array(
        [
            text_score,
            mbti_score,
            location_score,
        ],
        dtype=float,
    )

    # --------------------------------------------------------
    # Calculate predicted acceptance score
    # --------------------------------------------------------

    prediction = float(
        np.dot(weights, features)
    )

    prediction = float(
        np.clip(prediction, 0.0, 1.0)
    )

    # --------------------------------------------------------
    # Calculate feedback error
    # --------------------------------------------------------

    error = action - prediction

    # --------------------------------------------------------
    # Gradient-based weight update
    # --------------------------------------------------------

    updated = (
        weights
        + learning_rate * error * features
    )

    # --------------------------------------------------------
    # Keep weights valid
    # --------------------------------------------------------

    updated = project_to_simplex(updated)

    updated_weights = {
        key: float(value)
        for key, value in zip(WEIGHT_KEYS, updated)
    }

    # --------------------------------------------------------
    # Save updated weights
    # --------------------------------------------------------

    personalized_weights[user_id] = updated_weights

    save_personalized_weights(
        project_root,
        personalized_weights,
    )

    return updated_weights


# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    weights = load_personalized_weights(
        PROJECT_ROOT
    )

    print("\nADAPTIVE WEIGHT SYSTEM")
    print("=" * 60)

    print(
        f"Users with personalized weights: {len(weights)}"
    )

    if weights:
        example_user = next(iter(weights))

        print(
            f"\nExample user: {example_user}"
        )

        print(
            f"Weights: {weights[example_user]}"
        )

        print(
            f"Weight total: "
            f"{sum(weights[example_user].values()):.6f}"
        )
    else:
        print("No personalized weights found.")
