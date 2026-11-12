
"""
Intelligent Hybrid Recommendation System
Hybrid Profile Matching and Recommendation Engine
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.adaptive_updater import load_personalized_weights


# ============================================================
# Configuration
# ============================================================

DEFAULT_WEIGHTS = {
    "text": 1 / 3,
    "mbti": 1 / 3,
    "location": 1 / 3,
}

REQUIRED_USER_COLUMNS = {
    "user_id",
    "name",
    "profession",
    "location",
    "mbti",
}

REQUIRED_WEIGHT_KEYS = {
    "text",
    "mbti",
    "location",
}


# ============================================================
# Normalize User IDs
# ============================================================

def normalize_user_id(user_id):
    """Normalize user IDs for consistent matching."""

    if user_id is None:
        raise ValueError("User ID cannot be empty.")

    normalized = str(user_id).strip().upper()

    if not normalized:
        raise ValueError("User ID cannot be empty.")

    return normalized


# ============================================================
# Load Model Artifacts
# ============================================================

def load_artifacts(project_root):
    """
    Load all required recommendation artifacts.

    Returns:
        users_df,
        profile_embeddings,
        text_similarity_matrix,
        mbti_compatibility,
        personalized_weights
    """

    project_root = Path(project_root)

    data_dir = project_root / "data"
    models_dir = project_root / "models"

    required_files = {
        "User dataset": data_dir / "users.csv",
        "Profile embeddings": models_dir / "profile_embeddings.npy",
        "Text similarity matrix": models_dir / "text_similarity_matrix.npy",
        "MBTI compatibility": models_dir / "mbti_compatibility.joblib",
    }

    missing_files = [
        f"{name}: {path}"
        for name, path in required_files.items()
        if not path.is_file()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Required recommendation artifacts are missing:\n"
            + "\n".join(missing_files)
        )

    # --------------------------------------------------------
    # Load user dataset
    # --------------------------------------------------------

    try:
        users_df = pd.read_csv(data_dir / "users.csv")

    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as error:
        raise RuntimeError(
            f"Unable to load user dataset: {error}"
        ) from error

    if users_df.empty:
        raise ValueError("User dataset is empty.")

    users_df.columns = (
        users_df.columns.astype(str).str.strip()
    )

    missing_columns = (
        REQUIRED_USER_COLUMNS - set(users_df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing user columns: {sorted(missing_columns)}"
        )

    # Normalize profile identifiers and categorical values.
    users_df["user_id"] = (
        users_df["user_id"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    users_df["mbti"] = (
        users_df["mbti"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    users_df["location"] = (
        users_df["location"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    if users_df["user_id"].eq("").any():
        raise ValueError("Missing user IDs found.")

    if users_df["user_id"].duplicated().any():
        raise ValueError("Duplicate user IDs found.")

    if users_df["mbti"].eq("").any():
        raise ValueError("Missing MBTI values found.")

    # Matrix positions must correspond to dataframe positions.
    users_df = users_df.reset_index(drop=True)

    # --------------------------------------------------------
    # Load numerical artifacts
    # --------------------------------------------------------

    try:
        profile_embeddings = np.load(
            models_dir / "profile_embeddings.npy",
            allow_pickle=False,
        )

        text_similarity_matrix = np.load(
            models_dir / "text_similarity_matrix.npy",
            allow_pickle=False,
        )

        mbti_compatibility = joblib.load(
            models_dir / "mbti_compatibility.joblib"
        )

    except Exception as error:
        raise RuntimeError(
            f"Unable to load recommendation artifacts: {error}"
        ) from error

    # --------------------------------------------------------
    # Validate embeddings
    # --------------------------------------------------------

    if profile_embeddings.ndim != 2:
        raise ValueError(
            "Profile embeddings must be a 2D array."
        )

    if profile_embeddings.shape[0] != len(users_df):
        raise ValueError(
            "Profile embedding count does not match "
            "the number of users."
        )

    if not np.isfinite(profile_embeddings).all():
        raise ValueError(
            "Profile embeddings contain invalid values."
        )

    # --------------------------------------------------------
    # Validate similarity matrix
    # --------------------------------------------------------

    expected_shape = (
        len(users_df),
        len(users_df),
    )

    if text_similarity_matrix.shape != expected_shape:
        raise ValueError(
            "Similarity matrix dimensions do not match "
            f"the user dataset. Expected {expected_shape}, "
            f"received {text_similarity_matrix.shape}."
        )

    if not np.isfinite(text_similarity_matrix).all():
        raise ValueError(
            "Similarity matrix contains invalid values."
        )

    if (
        text_similarity_matrix.min() < -1.0001
        or text_similarity_matrix.max() > 1.0001
    ):
        raise ValueError(
            "Cosine similarity values must be between -1 and 1."
        )

    if not np.allclose(
        text_similarity_matrix,
        text_similarity_matrix.T,
        atol=1e-5,
    ):
        raise ValueError(
            "Text similarity matrix must be symmetric."
        )

    # --------------------------------------------------------
    # Validate MBTI compatibility matrix
    # --------------------------------------------------------

    if not isinstance(mbti_compatibility, dict):
        raise ValueError(
            "MBTI compatibility artifact must be a dictionary."
        )

    required_mbti_types = set(users_df["mbti"].unique())

    missing_mbti_types = (
        required_mbti_types - set(mbti_compatibility.keys())
    )

    if missing_mbti_types:
        raise ValueError(
            "Missing MBTI compatibility entries for: "
            f"{sorted(missing_mbti_types)}"
        )

    for viewer_mbti in required_mbti_types:

        if not isinstance(
            mbti_compatibility[viewer_mbti],
            dict,
        ):
            raise ValueError(
                f"Invalid MBTI compatibility row: {viewer_mbti}"
            )

        for candidate_mbti in required_mbti_types:

            if candidate_mbti not in mbti_compatibility[viewer_mbti]:
                raise ValueError(
                    "Missing MBTI compatibility score for "
                    f"{viewer_mbti} and {candidate_mbti}."
                )

            score = float(
                mbti_compatibility[viewer_mbti][candidate_mbti]
            )

            if not np.isfinite(score) or not 0 <= score <= 100:
                raise ValueError(
                    "Invalid MBTI compatibility score for "
                    f"{viewer_mbti} and {candidate_mbti}."
                )

    # --------------------------------------------------------
    # Load personalized weights
    # --------------------------------------------------------

    personalized_weights = load_personalized_weights(
        project_root
    )

    return (
        users_df,
        profile_embeddings,
        text_similarity_matrix,
        mbti_compatibility,
        personalized_weights,
    )


# ============================================================
# Location Compatibility
# ============================================================

def location_score(location_a, location_b):
    """
    Calculate location compatibility.

    Same location: 1.0
    Different location: 0.4
    """

    location_a = str(location_a).strip().casefold()
    location_b = str(location_b).strip().casefold()

    if location_a and location_a == location_b:
        return 1.0

    return 0.4


# ============================================================
# Get Valid Personalized Weights
# ============================================================

def get_user_weights(user_id, personalized_weights):
    """Return validated weights for a particular viewer."""

    normalized_id = normalize_user_id(user_id)

    weights = personalized_weights.get(
        normalized_id,
        DEFAULT_WEIGHTS.copy(),
    )

    if not isinstance(weights, dict):
        raise ValueError(
            f"Invalid personalized weights for {normalized_id}."
        )

    missing_keys = REQUIRED_WEIGHT_KEYS - set(weights.keys())

    if missing_keys:
        raise ValueError(
            f"Missing personalized weight keys: {sorted(missing_keys)}"
        )

    values = np.array(
        [
            weights["text"],
            weights["mbti"],
            weights["location"],
        ],
        dtype=float,
    )

    if not np.isfinite(values).all():
        raise ValueError(
            f"Invalid personalized weights for {normalized_id}."
        )

    if np.any(values < 0):
        raise ValueError(
            f"Negative personalized weights for {normalized_id}."
        )

    total = float(values.sum())

    if total <= 1e-12:
        raise ValueError(
            f"Personalized weights for {normalized_id} sum to zero."
        )

    values = values / total

    return {
        "text": float(values[0]),
        "mbti": float(values[1]),
        "location": float(values[2]),
    }


# ============================================================
# Calculate Individual Compatibility Components
# ============================================================

def calculate_match_components(
    viewer_index,
    candidate_index,
    users_df,
    text_similarity_matrix,
    mbti_compatibility,
    personalized_weights,
):
    """Calculate text, MBTI and location scores and weights."""

    if not 0 <= viewer_index < len(users_df):
        raise IndexError("Invalid viewer index.")

    if not 0 <= candidate_index < len(users_df):
        raise IndexError("Invalid candidate index.")

    viewer = users_df.iloc[viewer_index]
    candidate = users_df.iloc[candidate_index]

    # Text similarity
    text_score = float(
        text_similarity_matrix[
            viewer_index,
            candidate_index,
        ]
    )

    # Convert cosine similarity from [-1, 1] to [0, 1].
    text_score = float(
        np.clip((text_score + 1.0) / 2.0, 0.0, 1.0)
    )

    # MBTI compatibility
    viewer_mbti = str(viewer["mbti"]).strip().upper()
    candidate_mbti = str(candidate["mbti"]).strip().upper()

    try:
        mbti_score = float(
            mbti_compatibility[viewer_mbti][candidate_mbti]
        ) / 100.0

    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "Unable to calculate MBTI compatibility for "
            f"{viewer_mbti} and {candidate_mbti}."
        ) from error

    # Location compatibility
    loc_score = location_score(
        viewer["location"],
        candidate["location"],
    )

    # Personalized viewer weights
    weights = get_user_weights(
        viewer["user_id"],
        personalized_weights,
    )

    return {
        "text_score": text_score,
        "mbti_score": mbti_score,
        "location_score": loc_score,
        "text_weight": weights["text"],
        "mbti_weight": weights["mbti"],
        "location_weight": weights["location"],
    }


# ============================================================
# Calculate Hybrid Compatibility Score
# ============================================================

def calculate_score(
    viewer_index,
    candidate_index,
    users_df,
    text_similarity_matrix,
    mbti_compatibility,
    personalized_weights,
):
    """Calculate final weighted compatibility score."""

    components = calculate_match_components(
        viewer_index,
        candidate_index,
        users_df,
        text_similarity_matrix,
        mbti_compatibility,
        personalized_weights,
    )

    final_score = (
        components["text_weight"] * components["text_score"]
        + components["mbti_weight"] * components["mbti_score"]
        + components["location_weight"] * components["location_score"]
    )

    return float(
        np.clip(final_score, 0.0, 1.0)
    )


# ============================================================
# Explain Compatibility Score
# ============================================================

def get_match_explanation(
    viewer_index,
    candidate_index,
    users_df,
    text_similarity_matrix,
    mbti_compatibility,
    personalized_weights,
):
    """Return individual match components and their weights."""

    return calculate_match_components(
        viewer_index,
        candidate_index,
        users_df,
        text_similarity_matrix,
        mbti_compatibility,
        personalized_weights,
    )


# ============================================================
# Generate Top-K Recommendations
# ============================================================

def get_recommendations(
    user_id,
    users_df,
    text_similarity_matrix,
    mbti_compatibility,
    personalized_weights,
    top_k=5,
):
    """
    Generate top-K compatible profiles for an existing user.
    """

    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError(
            "top_k must be a positive integer."
        )

    required_columns = REQUIRED_USER_COLUMNS

    missing_columns = (
        required_columns - set(users_df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing user columns: {sorted(missing_columns)}"
        )

    users_df = users_df.reset_index(drop=True)

    if text_similarity_matrix.shape != (
        len(users_df),
        len(users_df),
    ):
        raise ValueError(
            "Similarity matrix and user dataset are misaligned."
        )

    normalized_user_id = normalize_user_id(user_id)

    normalized_ids = (
        users_df["user_id"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    matching_positions = np.flatnonzero(
        normalized_ids.to_numpy() == normalized_user_id
    )

    if len(matching_positions) == 0:
        raise ValueError(
            f"User ID '{normalized_user_id}' not found."
        )

    if len(matching_positions) > 1:
        raise ValueError(
            f"Duplicate user ID '{normalized_user_id}' found."
        )

    viewer_index = int(matching_positions[0])

    recommendations = []

    # Evaluate every candidate except the viewer.
    for candidate_index in range(len(users_df)):

        if candidate_index == viewer_index:
            continue

        score = calculate_score(
            viewer_index,
            candidate_index,
            users_df,
            text_similarity_matrix,
            mbti_compatibility,
            personalized_weights,
        )

        candidate = users_df.iloc[candidate_index]

        recommendations.append({
            "user_id": candidate["user_id"],
            "name": candidate["name"],
            "profession": candidate["profession"],
            "location": candidate["location"],
            "mbti": candidate["mbti"],
            "compatibility_score": round(
                score * 100,
                2,
            ),
        })

    if not recommendations:
        return pd.DataFrame(
            columns=[
                "user_id",
                "name",
                "profession",
                "location",
                "mbti",
                "compatibility_score",
            ]
        )

    recommendations_df = pd.DataFrame(
        recommendations
    )

    recommendations_df = (
        recommendations_df
        .sort_values(
            by=["compatibility_score", "user_id"],
            ascending=[False, True],
            kind="stable",
        )
        .head(top_k)
        .reset_index(drop=True)
    )

    return recommendations_df


# ============================================================
# Direct Two-User Compatibility
# ============================================================

def get_compatibility_score(
    user_id_a,
    user_id_b,
    project_root,
):
    """Calculate compatibility between two existing users."""

    user_id_a = normalize_user_id(user_id_a)
    user_id_b = normalize_user_id(user_id_b)

    if user_id_a == user_id_b:
        raise ValueError(
            "A user cannot be matched with themselves."
        )

    (
        users_df,
        profile_embeddings,
        text_similarity_matrix,
        mbti_compatibility,
        personalized_weights,
    ) = load_artifacts(project_root)

    normalized_ids = (
        users_df["user_id"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    rows_a = np.flatnonzero(
        normalized_ids.to_numpy() == user_id_a
    )

    rows_b = np.flatnonzero(
        normalized_ids.to_numpy() == user_id_b
    )

    if len(rows_a) == 0:
        raise ValueError(
            f"User ID '{user_id_a}' not found."
        )

    if len(rows_b) == 0:
        raise ValueError(
            f"User ID '{user_id_b}' not found."
        )

    score = calculate_score(
        int(rows_a[0]),
        int(rows_b[0]),
        users_df,
        text_similarity_matrix,
        mbti_compatibility,
        personalized_weights,
    )

    return round(score * 100, 2)


# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    (
        users_df,
        profile_embeddings,
        text_similarity_matrix,
        mbti_compatibility,
        personalized_weights,
    ) = load_artifacts(PROJECT_ROOT)

    print("\nHYBRID RECOMMENDATION ENGINE")
    print("=" * 60)

    print(f"Loaded profiles: {len(users_df)}")
    print(
        f"Embedding dimensions: {profile_embeddings.shape}"
    )
    print(
        f"Similarity matrix: {text_similarity_matrix.shape}"
    )
    print(
        f"Users with personalized weights: "
        f"{len(personalized_weights)}"
    )

    sample_user_id = users_df.iloc[0]["user_id"]

    print(f"\nSample viewer: {sample_user_id}")

    recommendations = get_recommendations(
        user_id=sample_user_id,
        users_df=users_df,
        text_similarity_matrix=text_similarity_matrix,
        mbti_compatibility=mbti_compatibility,
        personalized_weights=personalized_weights,
        top_k=5,
    )

    print("\nTOP 5 RECOMMENDATIONS")
    print("=" * 60)

    print(
        recommendations.to_string(index=False)
    )
