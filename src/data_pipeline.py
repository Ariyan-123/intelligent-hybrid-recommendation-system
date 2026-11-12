
"""
Intelligent Hybrid Recommendation System
Data Cleaning, Validation and Pipeline Execution
"""

from pathlib import Path

import pandas as pd


# ============================================================
# Configuration
# ============================================================

VALID_MBTI = {
    "INTJ", "INTP", "ENTJ", "ENTP",
    "INFJ", "INFP", "ENFJ", "ENFP",
    "ISTJ", "ISFJ", "ESTJ", "ESFJ",
    "ISTP", "ISFP", "ESTP", "ESFP",
}

REQUIRED_USER_COLUMNS = {
    "user_id",
    "name",
    "age",
    "location",
    "profession",
    "experience_years",
    "professional_summary",
    "about_me",
    "mbti",
    "interests",
}

REQUIRED_FEEDBACK_COLUMNS = {
    "user_id",
    "matched_user_id",
    "action",
    "timestamp",
}

MIN_FEEDBACK_INTERACTIONS = 5

AGE_MIN = 16
AGE_MAX = 100
EXPERIENCE_MIN = 0
EXPERIENCE_MAX = 80


# ============================================================
# Common Helpers
# ============================================================

def normalize_column_names(dataframe):
    """Remove surrounding whitespace from column names."""
    cleaned = dataframe.copy()
    cleaned.columns = cleaned.columns.astype(str).str.strip()
    return cleaned


def check_required_columns(dataframe, required_columns):
    """Return missing required columns."""
    return sorted(required_columns - set(dataframe.columns))


# ============================================================
# Clean User Profile Data
# ============================================================

def clean_users(users_df):
    """Clean and normalize user profile data."""

    cleaned = normalize_column_names(users_df)

    text_columns = [
        "user_id",
        "name",
        "location",
        "profession",
        "professional_summary",
        "about_me",
        "interests",
    ]

    for column in text_columns:
        if column in cleaned.columns:
            cleaned[column] = (
                cleaned[column]
                .fillna("")
                .astype(str)
                .str.strip()
            )

    if "user_id" in cleaned.columns:
        cleaned["user_id"] = cleaned["user_id"].str.upper()

    if "mbti" in cleaned.columns:
        cleaned["mbti"] = (
            cleaned["mbti"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.upper()
        )

    for column in ["age", "experience_years"]:
        if column in cleaned.columns:
            cleaned[column] = pd.to_numeric(
                cleaned[column],
                errors="coerce",
            )

    return cleaned


# ============================================================
# Clean Feedback Data
# ============================================================

def clean_feedback(feedback_df):
    """Clean and normalize feedback records."""

    cleaned = normalize_column_names(feedback_df)

    for column in ["user_id", "matched_user_id"]:
        if column in cleaned.columns:
            cleaned[column] = (
                cleaned[column]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.upper()
            )

    if "action" in cleaned.columns:
        cleaned["action"] = pd.to_numeric(
            cleaned["action"],
            errors="coerce",
        )

    if "timestamp" in cleaned.columns:
        cleaned["timestamp"] = pd.to_datetime(
            cleaned["timestamp"],
            errors="coerce",
        )

    return cleaned


# ============================================================
# Validate User Profiles
# ============================================================

def validate_users(users_df):
    """Validate user profile records and return error messages."""

    errors = []

    missing_columns = check_required_columns(
        users_df,
        REQUIRED_USER_COLUMNS,
    )

    if missing_columns:
        errors.append(
            f"Missing user columns: {missing_columns}"
        )
        return errors

    if users_df.empty:
        errors.append("User dataset is empty.")
        return errors

    # User ID validation
    if users_df["user_id"].eq("").any():
        errors.append("Missing or blank user_id values found.")

    if users_df["user_id"].duplicated().any():
        errors.append("Duplicate user_id values found.")

    # Required text validation
    required_text_fields = [
        "name",
        "location",
        "profession",
        "professional_summary",
        "about_me",
        "interests",
    ]

    for column in required_text_fields:
        empty_values = (
            users_df[column]
            .fillna("")
            .astype(str)
            .str.strip()
            .eq("")
        )

        if empty_values.any():
            errors.append(
                f"Missing or blank values found in '{column}'."
            )

    # MBTI validation
    invalid_mbti = (
        set(users_df["mbti"].dropna().astype(str).str.upper())
        - VALID_MBTI
    )

    if invalid_mbti:
        errors.append(
            f"Invalid MBTI values: {sorted(invalid_mbti)}"
        )

    if users_df["mbti"].isna().any() or users_df["mbti"].eq("").any():
        errors.append("Missing or blank MBTI values found.")

    # Age validation
    age = users_df["age"]

    if age.isna().any():
        errors.append("Missing or invalid age values found.")
    else:
        invalid_age = ~age.between(AGE_MIN, AGE_MAX)

        if invalid_age.any():
            errors.append(
                f"Age must be between {AGE_MIN} and {AGE_MAX}."
            )

    # Experience validation
    experience = users_df["experience_years"]

    if experience.isna().any():
        errors.append(
            "Missing or invalid experience_years values found."
        )
    else:
        invalid_experience = ~experience.between(
            EXPERIENCE_MIN,
            EXPERIENCE_MAX,
        )

        if invalid_experience.any():
            errors.append(
                "Experience must be between "
                f"{EXPERIENCE_MIN} and {EXPERIENCE_MAX} years."
            )

    # Duplicate complete rows
    if users_df.duplicated().any():
        errors.append("Duplicate complete user rows found.")

    return errors


# ============================================================
# Validate Feedback Data
# ============================================================

def validate_feedback(feedback_df, users_df):
    """Validate feedback records against existing user profiles."""

    errors = []

    missing_columns = check_required_columns(
        feedback_df,
        REQUIRED_FEEDBACK_COLUMNS,
    )

    if missing_columns:
        errors.append(
            f"Missing feedback columns: {missing_columns}"
        )
        return errors

    if feedback_df.empty:
        errors.append("Feedback dataset is empty.")
        return errors

    if "user_id" not in users_df.columns:
        errors.append(
            "Cannot validate feedback: user dataset has no user_id column."
        )
        return errors

    valid_users = set(users_df["user_id"].dropna())

    # Check missing IDs
    for column in ["user_id", "matched_user_id"]:
        if feedback_df[column].isna().any():
            errors.append(
                f"Missing values found in feedback '{column}'."
            )

        if feedback_df[column].eq("").any():
            errors.append(
                f"Blank values found in feedback '{column}'."
            )

    # Check unknown users
    invalid_viewers = (
        set(feedback_df["user_id"].dropna()) - valid_users
    )

    if invalid_viewers:
        errors.append(
            f"Unknown feedback user IDs: {sorted(invalid_viewers)}"
        )

    invalid_matches = (
        set(feedback_df["matched_user_id"].dropna()) - valid_users
    )

    if invalid_matches:
        errors.append(
            f"Unknown matched user IDs: {sorted(invalid_matches)}"
        )

    # Action validation
    invalid_actions = (
        set(feedback_df["action"].dropna()) - {0, 1}
    )

    if invalid_actions:
        errors.append(
            f"Invalid feedback action values: {sorted(invalid_actions)}"
        )

    if feedback_df["action"].isna().any():
        errors.append("Missing or invalid feedback action values found.")

    # Timestamp validation
    if feedback_df["timestamp"].isna().any():
        errors.append(
            "Missing or invalid feedback timestamps found."
        )

    # Self-matching validation
    self_matches = (
        feedback_df["user_id"]
        == feedback_df["matched_user_id"]
    )

    if self_matches.any():
        errors.append("Self-matching feedback records found.")

    # Duplicate complete feedback records
    if feedback_df.duplicated().any():
        errors.append("Duplicate complete feedback records found.")

    return errors


# ============================================================
# Feedback Interaction Warnings
# ============================================================

def get_feedback_warnings(feedback_df, users_df):
    """
    Return warnings for users with fewer than the recommended
    minimum number of feedback interactions.
    """

    warnings = []

    if feedback_df.empty or "user_id" not in feedback_df.columns:
        return warnings

    interaction_counts = feedback_df.groupby("user_id").size()

    counts = (
        interaction_counts
        .reindex(users_df["user_id"], fill_value=0)
    )

    affected_users = counts[
        counts < MIN_FEEDBACK_INTERACTIONS
    ].index.tolist()

    if affected_users:
        warnings.append(
            f"{len(affected_users)} users have fewer than "
            f"{MIN_FEEDBACK_INTERACTIONS} feedback interactions."
        )

    return warnings


# ============================================================
# Run Data Pipeline
# ============================================================

def run_pipeline(project_root):
    """Load, clean and validate project datasets."""

    project_root = Path(project_root)

    users_path = project_root / "data" / "users.csv"
    feedback_path = project_root / "data" / "feedback.csv"

    print("\nDATA PIPELINE")
    print("=" * 60)

    # Check file availability
    missing_files = [
        path for path in [users_path, feedback_path]
        if not path.is_file()
    ]

    if missing_files:
        print("\nVALIDATION FAILED")

        for path in missing_files:
            print(f"- Required file not found: {path}")

        return False

    # Load datasets
    try:
        users_raw = pd.read_csv(users_path)
        feedback_raw = pd.read_csv(feedback_path)

    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as error:
        print("\nDATA LOADING FAILED")
        print(f"- {error}")
        return False

    # Cleaning
    users_df = clean_users(users_raw)
    feedback_df = clean_feedback(feedback_raw)

    # Validation
    user_errors = validate_users(users_df)
    feedback_errors = validate_feedback(
        feedback_df,
        users_df,
    )

    errors = user_errors + feedback_errors
    warnings = get_feedback_warnings(
        feedback_df,
        users_df,
    )

    # Pipeline report
    print("\nDATASET SUMMARY")
    print("=" * 60)

    print(f"Total users: {len(users_df)}")
    print(f"Total feedback records: {len(feedback_df)}")

    if "user_id" in users_df.columns:
        print(
            f"Unique users: {users_df['user_id'].nunique()}"
        )

    if "action" in feedback_df.columns:
        print(
            "Accepted recommendations: "
            f"{int((feedback_df['action'] == 1).sum())}"
        )
        print(
            "Rejected recommendations: "
            f"{int((feedback_df['action'] == 0).sum())}"
        )

    print("\nVALIDATION REPORT")
    print("=" * 60)

    if warnings:
        print("\nWARNINGS")

        for warning in warnings:
            print(f"- {warning}")

    if errors:
        print("\nVALIDATION FAILED")

        for error in errors:
            print(f"- {error}")

        return False

    print("\nVALIDATION PASSED")
    print("Both datasets passed the required validation checks.")

    return True


# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    success = run_pipeline(PROJECT_ROOT)

    raise SystemExit(0 if success else 1)
