
"""
Intelligent Hybrid Recommendation System
Streamlit Application

Features:
- Existing profile recommendations
- New profile recommendations
- Semantic text similarity
- MBTI compatibility
- Location compatibility
- Personalized adaptive weights
- Accept/reject feedback
"""

from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.matcher import (
    load_artifacts,
    get_recommendations,
    get_match_explanation,
)

from src.nlp_pipeline import preprocess_text

from src.adaptive_updater import update_user_weights


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

FEEDBACK_PATH = DATA_DIR / "ui_feedback.csv"

MBTI_TYPES = [
    "INTJ", "INTP", "ENTJ", "ENTP",
    "INFJ", "INFP", "ENFJ", "ENFP",
    "ISTJ", "ISFJ", "ESTJ", "ESFJ",
    "ISTP", "ISFP", "ESTP", "ESFP",
]

DEFAULT_WEIGHTS = {
    "text": 1 / 3,
    "mbti": 1 / 3,
    "location": 1 / 3,
}


# ============================================================
# 2. STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Intelligent Hybrid Recommendation System",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 3. LOAD PROJECT ARTIFACTS
# ============================================================

@st.cache_resource
def load_project_artifacts():

    return load_artifacts(
        str(PROJECT_ROOT)
    )


@st.cache_resource
def load_embedding_model():

    model_name_path = (
        MODELS_DIR / "embedding_model_name.joblib"
    )

    if model_name_path.exists():

        model_name = joblib.load(
            model_name_path
        )

    else:

        model_name = "all-MiniLM-L6-v2"

    if not isinstance(model_name, str) or not model_name.strip():

        raise ValueError(
            "Invalid embedding model configuration."
        )

    return SentenceTransformer(
        model_name.strip()
    )


def load_initial_weights():

    weights_path = (
        MODELS_DIR / "initial_weights.joblib"
    )

    if weights_path.exists():

        weights = joblib.load(
            weights_path
        )

    else:

        weights = {
            "text": 0.5,
            "mbti": 0.3,
            "location": 0.2,
        }

    required_keys = {
        "text",
        "mbti",
        "location",
    }

    if not isinstance(weights, dict):

        raise ValueError(
            "Initial weights must be a dictionary."
        )

    if not required_keys.issubset(weights):

        raise ValueError(
            "Initial weights are missing required components."
        )

    values = np.array(
        [
            weights["text"],
            weights["mbti"],
            weights["location"],
        ],
        dtype=float,
    )

    if (
        not np.isfinite(values).all()
        or np.any(values < 0)
        or values.sum() <= 0
    ):

        raise ValueError(
            "Initial weights contain invalid values."
        )

    values = values / values.sum()

    return {
        "text": float(values[0]),
        "mbti": float(values[1]),
        "location": float(values[2]),
    }


try:

    (
        users_df,
        profile_embeddings,
        text_similarity_matrix,
        mbti_compatibility,
        personalized_weights,
    ) = load_project_artifacts()

    embedding_model = load_embedding_model()

    initial_weights = load_initial_weights()

except Exception as error:

    st.error(
        f"Unable to initialize the recommendation system: {error}"
    )

    st.stop()


# ============================================================
# 4. SESSION STATE
# ============================================================

SESSION_DEFAULTS = {
    "recommendations": None,
    "recommendation_user_id": None,
    "new_recommendations": None,
    "new_profile": None,
}

for key, value in SESSION_DEFAULTS.items():

    if key not in st.session_state:

        st.session_state[key] = value


# ============================================================
# 5. FEEDBACK DATA MANAGEMENT
# ============================================================

FEEDBACK_COLUMNS = [
    "user_id",
    "matched_user_id",
    "action",
    "timestamp",
]


def load_ui_feedback():

    if not FEEDBACK_PATH.exists():

        return pd.DataFrame(
            columns=FEEDBACK_COLUMNS
        )

    try:

        feedback_df = pd.read_csv(
            FEEDBACK_PATH
        )

        missing_columns = (
            set(FEEDBACK_COLUMNS)
            - set(feedback_df.columns)
        )

        if missing_columns:

            raise ValueError(
                f"Missing feedback columns: {missing_columns}"
            )

        feedback_df = feedback_df[
            FEEDBACK_COLUMNS
        ].copy()

        feedback_df["user_id"] = (
            feedback_df["user_id"]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        feedback_df["matched_user_id"] = (
            feedback_df["matched_user_id"]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        feedback_df["action"] = pd.to_numeric(
            feedback_df["action"],
            errors="coerce",
        )

        feedback_df["timestamp"] = pd.to_datetime(
            feedback_df["timestamp"],
            errors="coerce",
        )

        feedback_df = feedback_df.dropna(
            subset=FEEDBACK_COLUMNS
        )

        feedback_df = feedback_df[
            feedback_df["action"].isin([0, 1])
        ]

        return feedback_df.reset_index(
            drop=True
        )

    except Exception as error:

        st.error(
            f"Unable to read application feedback: {error}"
        )

        return pd.DataFrame(
            columns=FEEDBACK_COLUMNS
        )


def get_existing_feedback(
    user_id,
    matched_user_id,
):

    feedback_df = load_ui_feedback()

    if feedback_df.empty:

        return None

    user_id = str(user_id).strip().upper()

    matched_user_id = (
        str(matched_user_id).strip().upper()
    )

    matching_feedback = feedback_df[
        (
            feedback_df["user_id"] == user_id
        )
        &
        (
            feedback_df["matched_user_id"]
            == matched_user_id
        )
    ]

    if matching_feedback.empty:

        return None

    latest_feedback = (
        matching_feedback
        .sort_values("timestamp")
        .iloc[-1]
    )

    return int(
        latest_feedback["action"]
    )


def save_feedback(
    user_id,
    matched_user_id,
    action,
):

    if action not in [0, 1]:

        raise ValueError(
            "Feedback action must be 0 or 1."
        )

    user_id = str(user_id).strip().upper()

    matched_user_id = (
        str(matched_user_id).strip().upper()
    )

    if user_id == matched_user_id:

        raise ValueError(
            "A user cannot provide feedback on their own profile."
        )

    existing_action = get_existing_feedback(
        user_id,
        matched_user_id,
    )

    if existing_action is not None:

        return False

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    new_feedback = pd.DataFrame([
        {
            "user_id": user_id,
            "matched_user_id": matched_user_id,
            "action": int(action),
            "timestamp": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
        }
    ])

    new_feedback.to_csv(
        FEEDBACK_PATH,
        mode="a",
        header=not FEEDBACK_PATH.exists(),
        index=False,
    )

    return True


# ============================================================
# 6. LOCATION COMPATIBILITY
# ============================================================

def calculate_location_score(
    location_a,
    location_b,
):

    location_a = str(
        location_a
    ).strip().casefold()

    location_b = str(
        location_b
    ).strip().casefold()

    if location_a == location_b:

        return 1.0

    return 0.4


# ============================================================
# 7. NEW PROFILE RECOMMENDATION ENGINE
# ============================================================

def get_new_profile_recommendations(
    profile,
    top_k=5,
):

    if top_k < 1:

        raise ValueError(
            "top_k must be at least 1."
        )

    mbti_type = str(
        profile["mbti"]
    ).strip().upper()

    if mbti_type not in MBTI_TYPES:

        raise ValueError(
            "Invalid MBTI personality type."
        )

    if mbti_type not in mbti_compatibility:

        raise ValueError(
            f"MBTI compatibility data not found for {mbti_type}."
        )

    # --------------------------------------------------------
    # Combine professional and personal information
    # --------------------------------------------------------

    raw_text = (
        str(profile["professional_summary"])
        + " "
        + str(profile["about_me"])
    )

    processed_text = preprocess_text(
        raw_text
    )

    if not processed_text.strip():

        raise ValueError(
            "The entered profile does not contain usable text."
        )

    # --------------------------------------------------------
    # Generate normalized embedding
    # --------------------------------------------------------

    new_embedding = embedding_model.encode(
        [processed_text],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    if new_embedding.shape[1] != profile_embeddings.shape[1]:

        raise ValueError(
            "New profile embedding dimensions do not match "
            "the stored profile embeddings."
        )

    # --------------------------------------------------------
    # Calculate semantic similarity
    # --------------------------------------------------------

    cosine_scores = cosine_similarity(
        new_embedding,
        profile_embeddings,
    )[0]

    # Matcher's standard normalization:
    # cosine similarity [-1, 1] -> compatibility [0, 1]
    similarity_scores = np.clip(
        (cosine_scores + 1.0) / 2.0,
        0.0,
        1.0,
    )

    recommendations = []

    # --------------------------------------------------------
    # Calculate hybrid compatibility
    # --------------------------------------------------------

    for candidate_index, candidate in (
        users_df.reset_index(drop=True).iterrows()
    ):

        candidate_mbti = str(
            candidate["mbti"]
        ).strip().upper()

        if candidate_mbti not in mbti_compatibility[mbti_type]:

            raise ValueError(
                f"Missing MBTI compatibility score for "
                f"{mbti_type} and {candidate_mbti}."
            )

        text_score = float(
            similarity_scores[candidate_index]
        )

        mbti_score = float(
            mbti_compatibility[mbti_type][candidate_mbti]
        ) / 100.0

        location_score = calculate_location_score(
            profile["location"],
            candidate["location"],
        )

        # New profiles do not have interaction history.
        # Use the configured initial weights.
        weights = initial_weights

        final_score = (
            weights["text"] * text_score
            + weights["mbti"] * mbti_score
            + weights["location"] * location_score
        )

        recommendations.append(
            {
                "user_id": candidate["user_id"],
                "name": candidate["name"],
                "profession": candidate["profession"],
                "location": candidate["location"],
                "mbti": candidate["mbti"],
                "compatibility_score": round(
                    float(np.clip(final_score, 0, 1)) * 100,
                    2,
                ),
                "text_score": text_score,
                "mbti_score": mbti_score,
                "location_score": location_score,
                "text_weight": weights["text"],
                "mbti_weight": weights["mbti"],
                "location_weight": weights["location"],
            }
        )

    recommendations_df = pd.DataFrame(
        recommendations
    )

    return (
        recommendations_df
        .sort_values(
            "compatibility_score",
            ascending=False,
            kind="stable",
        )
        .head(top_k)
        .reset_index(drop=True)
    )


# ============================================================
# 8. UPDATE PERSONALIZED WEIGHTS
# ============================================================

def process_feedback(
    user_id,
    match,
    explanation,
    action,
):

    saved = save_feedback(
        user_id=user_id,
        matched_user_id=match["user_id"],
        action=action,
    )

    if not saved:

        return False, None

    try:

        updated_weights = update_user_weights(
            project_root=str(PROJECT_ROOT),
            user_id=user_id,
            text_score=float(
                explanation["text_score"]
            ),
            mbti_score=float(
                explanation["mbti_score"]
            ),
            location_score=float(
                explanation["location_score"]
            ),
            action=action,
        )

        # Update the application's in-memory weights too.
        personalized_weights[
            str(user_id).strip().upper()
        ] = updated_weights

        return True, updated_weights

    except Exception as error:

        # Feedback has already been saved.
        # Do not pretend the weight update succeeded.
        return True, None


# ============================================================
# 9. APPLICATION HEADER
# ============================================================

st.title(
    "🔗 Intelligent Hybrid Recommendation System"
)

st.caption(
    "AI-powered professional profile matching using "
    "semantic similarity, MBTI compatibility, "
    "location matching and adaptive personalized weights."
)

st.divider()


# ============================================================
# 10. PROFILE MODE
# ============================================================

st.subheader(
    "Choose Profile Mode"
)

profile_mode = st.radio(
    "How would you like to find compatible profiles?",
    [
        "Existing Profile",
        "Enter New Profile",
    ],
    horizontal=True,
)


# ============================================================
# 11. EXISTING PROFILE MODE
# ============================================================

if profile_mode == "Existing Profile":

    st.subheader(
        "Select Your Profile"
    )

    user_options = (
        users_df["user_id"]
        .astype(str)
        .tolist()
    )

    user_id = st.selectbox(
        "Select your profile",
        options=user_options,
    )

    # --------------------------------------------------------
    # Clear recommendations when selected user changes
    # --------------------------------------------------------

    if (
        st.session_state.recommendation_user_id
        != user_id
    ):

        st.session_state.recommendations = None

        st.session_state.recommendation_user_id = (
            user_id
        )

    selected_user = users_df[
        users_df["user_id"].astype(str)
        == str(user_id)
    ].iloc[0]

    # --------------------------------------------------------
    # Display profile information
    # --------------------------------------------------------

    st.subheader(
        "Your Profile"
    )

    profile_col1, profile_col2, profile_col3 = (
        st.columns(3)
    )

    with profile_col1:

        st.write(
            f"**Name:** {selected_user['name']}"
        )

        st.write(
            f"**Profession:** {selected_user['profession']}"
        )

    with profile_col2:

        st.write(
            f"**Location:** {selected_user['location']}"
        )

        st.write(
            f"**Experience:** "
            f"{selected_user['experience_years']} years"
        )

    with profile_col3:

        st.write(
            f"**MBTI:** {selected_user['mbti']}"
        )

        st.write(
            f"**Interests:** {selected_user['interests']}"
        )

    # --------------------------------------------------------
    # Show current personalized weights
    # --------------------------------------------------------

    current_weights = personalized_weights.get(
        str(user_id).strip().upper(),
        DEFAULT_WEIGHTS,
    )

    with st.expander(
        "View Your Matching Weights"
    ):

        weight_col1, weight_col2, weight_col3 = (
            st.columns(3)
        )

        with weight_col1:

            st.metric(
                "Text",
                f"{current_weights['text'] * 100:.2f}%",
            )

        with weight_col2:

            st.metric(
                "MBTI",
                f"{current_weights['mbti'] * 100:.2f}%",
            )

        with weight_col3:

            st.metric(
                "Location",
                f"{current_weights['location'] * 100:.2f}%",
            )

    # --------------------------------------------------------
    # Generate recommendations
    # --------------------------------------------------------

    if st.button(
        "Find Top 5 Matches",
        type="primary",
        key="existing_profile_button",
    ):

        try:

            with st.spinner(
                "Finding compatible profiles..."
            ):

                st.session_state.recommendations = (
                    get_recommendations(
                        user_id=user_id,
                        users_df=users_df,
                        text_similarity_matrix=(
                            text_similarity_matrix
                        ),
                        mbti_compatibility=(
                            mbti_compatibility
                        ),
                        personalized_weights=(
                            personalized_weights
                        ),
                        top_k=5,
                    )
                )

                st.session_state.recommendation_user_id = (
                    user_id
                )

        except Exception as error:

            st.error(
                f"Unable to generate recommendations: {error}"
            )

    # --------------------------------------------------------
    # Display recommendations
    # --------------------------------------------------------

    if (
        st.session_state.recommendations is not None
        and st.session_state.recommendation_user_id == user_id
    ):

        recommendations = (
            st.session_state.recommendations
        )

        if recommendations.empty:

            st.warning(
                "No compatible profiles were found."
            )

        else:

            st.success(
                f"Top {len(recommendations)} matches generated."
            )

            st.subheader(
                "Top 5 Compatible Profiles"
            )

            viewer_index = users_df[
                users_df["user_id"].astype(str)
                == str(user_id)
            ].index[0]

            # ------------------------------------------------
            # Recommendation cards
            # ------------------------------------------------

            for rank, (_, match) in enumerate(
                recommendations.iterrows(),
                start=1,
            ):

                candidate_index = users_df[
                    users_df["user_id"].astype(str)
                    == str(match["user_id"])
                ].index[0]

                explanation = get_match_explanation(
                    viewer_index=viewer_index,
                    candidate_index=candidate_index,
                    users_df=users_df,
                    text_similarity_matrix=(
                        text_similarity_matrix
                    ),
                    mbti_compatibility=(
                        mbti_compatibility
                    ),
                    personalized_weights=(
                        personalized_weights
                    ),
                )

                st.markdown(
                    f"""
### #{rank} {match['name']} — {match['user_id']}

**{match['profession']}** • {match['location']} • MBTI: **{match['mbti']}**

**Compatibility: {match['compatibility_score']:.2f}%**
"""
                )

                st.progress(
                    float(
                        np.clip(
                            match["compatibility_score"] / 100,
                            0,
                            1,
                        )
                    )
                )

                # --------------------------------------------
                # Match explanation
                # --------------------------------------------

                with st.expander(
                    "Why this profile matches"
                ):

                    explanation_col1, explanation_col2, explanation_col3 = (
                        st.columns(3)
                    )

                    with explanation_col1:

                        st.metric(
                            "Text Similarity",
                            f"{explanation['text_score'] * 100:.2f}%",
                        )

                        st.caption(
                            f"Weight: "
                            f"{explanation['text_weight'] * 100:.2f}%"
                        )

                    with explanation_col2:

                        st.metric(
                            "MBTI Compatibility",
                            f"{explanation['mbti_score'] * 100:.2f}%",
                        )

                        st.caption(
                            f"Weight: "
                            f"{explanation['mbti_weight'] * 100:.2f}%"
                        )

                    with explanation_col3:

                        st.metric(
                            "Location Compatibility",
                            f"{explanation['location_score'] * 100:.2f}%",
                        )

                        st.caption(
                            f"Weight: "
                            f"{explanation['location_weight'] * 100:.2f}%"
                        )

                # --------------------------------------------
                # Existing feedback
                # --------------------------------------------

                existing_action = get_existing_feedback(
                    user_id=user_id,
                    matched_user_id=match["user_id"],
                )

                if existing_action == 1:

                    st.success(
                        "✓ You already accepted this profile."
                    )

                elif existing_action == 0:

                    st.error(
                        "✕ You already rejected this profile."
                    )

                else:

                    feedback_col1, feedback_col2 = (
                        st.columns(2)
                    )

                    with feedback_col1:

                        if st.button(
                            "👍 Accept",
                            key=f"accept_{user_id}_{match['user_id']}",
                        ):

                            try:

                                saved, updated_weights = process_feedback(
                                    user_id,
                                    match,
                                    explanation,
                                    action=1,
                                )

                                if saved:

                                    st.success(
                                        "Feedback recorded: Accepted."
                                    )

                                    if updated_weights is not None:

                                        st.info(
                                            "Personalized weights updated: "
                                            f"Text {updated_weights['text'] * 100:.2f}% | "
                                            f"MBTI {updated_weights['mbti'] * 100:.2f}% | "
                                            f"Location {updated_weights['location'] * 100:.2f}%"
                                        )

                                    else:

                                        st.warning(
                                            "Feedback was saved, but the "
                                            "weight update could not be completed."
                                        )

                                    st.rerun()

                                else:

                                    st.info(
                                        "Feedback for this profile "
                                        "has already been recorded."
                                    )

                            except Exception as error:

                                st.error(
                                    f"Unable to save feedback: {error}"
                                )

                    with feedback_col2:

                        if st.button(
                            "👎 Reject",
                            key=f"reject_{user_id}_{match['user_id']}",
                        ):

                            try:

                                saved, updated_weights = process_feedback(
                                    user_id,
                                    match,
                                    explanation,
                                    action=0,
                                )

                                if saved:

                                    st.warning(
                                        "Feedback recorded: Rejected."
                                    )

                                    if updated_weights is not None:

                                        st.info(
                                            "Personalized weights updated: "
                                            f"Text {updated_weights['text'] * 100:.2f}% | "
                                            f"MBTI {updated_weights['mbti'] * 100:.2f}% | "
                                            f"Location {updated_weights['location'] * 100:.2f}%"
                                        )

                                    else:

                                        st.warning(
                                            "Feedback was saved, but the "
                                            "weight update could not be completed."
                                        )

                                    st.rerun()

                                else:

                                    st.info(
                                        "Feedback for this profile "
                                        "has already been recorded."
                                    )

                            except Exception as error:

                                st.error(
                                    f"Unable to save feedback: {error}"
                                )

                if rank < len(recommendations):

                    st.divider()


# ============================================================
# 12. NEW PROFILE MODE
# ============================================================

else:

    st.subheader(
        "Enter Your Profile"
    )

    st.caption(
        "Provide your professional and personal information "
        "to discover compatible profiles."
    )

    # --------------------------------------------------------
    # Basic information
    # --------------------------------------------------------

    input_col1, input_col2 = st.columns(2)

    with input_col1:

        new_name = st.text_input(
            "Name",
            placeholder="Enter your name",
        )

        new_profession = st.text_input(
            "Profession",
            placeholder="e.g. Data Scientist",
        )

        new_location = st.text_input(
            "Location",
            placeholder="e.g. Bhubaneswar",
        )

    with input_col2:

        new_age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=21,
            step=1,
        )

        new_experience = st.number_input(
            "Experience (years)",
            min_value=0,
            max_value=50,
            value=0,
            step=1,
        )

        new_mbti = st.selectbox(
            "MBTI Personality Type",
            options=MBTI_TYPES,
        )

    # --------------------------------------------------------
    # Professional information
    # --------------------------------------------------------

    new_professional_summary = st.text_area(
        "Professional Summary",
        placeholder=(
            "Describe your professional experience, skills, "
            "career goals and strengths."
        ),
        height=140,
    )

    new_about_me = st.text_area(
        "About Me",
        placeholder=(
            "Describe your personality, interests, "
            "working style and personal preferences."
        ),
        height=140,
    )

    new_interests = st.text_input(
        "Interests",
        placeholder="e.g. AI, analytics, research, technology",
    )

    # --------------------------------------------------------
    # Generate new profile recommendations
    # --------------------------------------------------------

    if st.button(
        "Find Top 5 Matches",
        type="primary",
        key="new_profile_button",
    ):

        required_fields = {
            "Name": new_name,
            "Profession": new_profession,
            "Location": new_location,
            "Professional Summary": new_professional_summary,
            "About Me": new_about_me,
            "Interests": new_interests,
        }

        missing_fields = [
            field_name
            for field_name, value in required_fields.items()
            if not str(value).strip()
        ]

        if missing_fields:

            st.error(
                "Please complete: "
                + ", ".join(missing_fields)
            )

        else:

            new_profile = {
                "name": new_name.strip(),
                "age": int(new_age),
                "location": new_location.strip(),
                "profession": new_profession.strip(),
                "experience_years": int(new_experience),
                "professional_summary": (
                    new_professional_summary.strip()
                ),
                "about_me": new_about_me.strip(),
                "mbti": new_mbti,
                "interests": new_interests.strip(),
            }

            try:

                with st.spinner(
                    "Analyzing your profile and finding matches..."
                ):

                    new_recommendations = (
                        get_new_profile_recommendations(
                            new_profile,
                            top_k=5,
                        )
                    )

                st.session_state.new_profile = (
                    new_profile
                )

                st.session_state.new_recommendations = (
                    new_recommendations
                )

            except Exception as error:

                st.error(
                    f"Unable to generate matches: {error}"
                )

    # --------------------------------------------------------
    # Display new profile recommendations
    # --------------------------------------------------------

    if st.session_state.new_recommendations is not None:

        recommendations = (
            st.session_state.new_recommendations
        )

        profile = (
            st.session_state.new_profile
        )

        st.success(
            "Top matches generated for your profile."
        )

        st.subheader(
            "Your Profile"
        )

        profile_col1, profile_col2, profile_col3 = (
            st.columns(3)
        )

        with profile_col1:

            st.write(
                f"**Name:** {profile['name']}"
            )

            st.write(
                f"**Profession:** {profile['profession']}"
            )

        with profile_col2:

            st.write(
                f"**Location:** {profile['location']}"
            )

            st.write(
                f"**Experience:** {profile['experience_years']} years"
            )

        with profile_col3:

            st.write(
                f"**MBTI:** {profile['mbti']}"
            )

            st.write(
                f"**Interests:** {profile['interests']}"
            )

        st.subheader(
            "Top 5 Compatible Profiles"
        )

        for rank, (_, match) in enumerate(
            recommendations.iterrows(),
            start=1,
        ):

            st.markdown(
                f"""
### #{rank} {match['name']} — {match['user_id']}

**{match['profession']}** • {match['location']} • MBTI: **{match['mbti']}**

**Compatibility: {match['compatibility_score']:.2f}%**
"""
            )

            st.progress(
                float(
                    np.clip(
                        match["compatibility_score"] / 100,
                        0,
                        1,
                    )
                )
            )

            with st.expander(
                "Why this profile matches"
            ):

                explanation_col1, explanation_col2, explanation_col3 = (
                    st.columns(3)
                )

                with explanation_col1:

                    st.metric(
                        "Text Similarity",
                        f"{match['text_score'] * 100:.2f}%",
                    )

                    st.caption(
                        f"Weight: {match['text_weight'] * 100:.2f}%"
                    )

                with explanation_col2:

                    st.metric(
                        "MBTI Compatibility",
                        f"{match['mbti_score'] * 100:.2f}%",
                    )

                    st.caption(
                        f"Weight: {match['mbti_weight'] * 100:.2f}%"
                    )

                with explanation_col3:

                    st.metric(
                        "Location Compatibility",
                        f"{match['location_score'] * 100:.2f}%",
                    )

                    st.caption(
                        f"Weight: {match['location_weight'] * 100:.2f}%"
                    )

            if rank < len(recommendations):

                st.divider()


# ============================================================
# 13. FEEDBACK ACTIVITY SUMMARY
# ============================================================

if profile_mode == "Existing Profile":

    feedback_df = load_ui_feedback()

    if not feedback_df.empty:

        user_feedback = feedback_df[
            feedback_df["user_id"]
            == str(user_id).strip().upper()
        ]

        if not user_feedback.empty:

            accepted_count = int(
                (user_feedback["action"] == 1).sum()
            )

            rejected_count = int(
                (user_feedback["action"] == 0).sum()
            )

            total_feedback = len(
                user_feedback
            )

            st.divider()

            st.subheader(
                "Your Feedback Activity"
            )

            feedback_col1, feedback_col2, feedback_col3 = (
                st.columns(3)
            )

            with feedback_col1:

                st.metric(
                    "Accepted",
                    accepted_count,
                )

            with feedback_col2:

                st.metric(
                    "Rejected",
                    rejected_count,
                )

            with feedback_col3:

                st.metric(
                    "Total Feedback",
                    total_feedback,
                )


# ============================================================
# 14. FOOTER
# ============================================================

st.divider()

st.caption(
    "Intelligent Hybrid Recommendation System | "
    "Semantic Matching • Personality Compatibility • "
    "Adaptive Learning"
)
