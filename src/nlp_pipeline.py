
"""
Intelligent Hybrid Recommendation System
NLP Preprocessing and Semantic Analysis Pipeline
"""

from pathlib import Path
import re
import warnings

import joblib
import nltk
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# Configuration
# ============================================================

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"

REQUIRED_COLUMNS = {
    "user_id",
    "professional_summary",
    "about_me",
}

EXPECTED_EMBEDDING_DIMENSION = 384

NLTK_RESOURCES = [
    ("corpora/stopwords", "stopwords"),
    ("corpora/wordnet", "wordnet"),
    ("corpora/omw-1.4", "omw-1.4"),
]


# ============================================================
# NLTK Resource Management
# ============================================================

def download_nltk_resources():
    """Download required NLTK resources when unavailable."""

    for resource_path, resource_name in NLTK_RESOURCES:

        try:
            nltk.data.find(resource_path)

        except LookupError:
            try:
                downloaded = nltk.download(
                    resource_name,
                    quiet=True,
                )

                if not downloaded:
                    raise RuntimeError(
                        f"Unable to download NLTK resource: "
                        f"{resource_name}"
                    )

            except Exception as error:
                raise RuntimeError(
                    f"NLTK resource setup failed for "
                    f"'{resource_name}': {error}"
                ) from error


# ============================================================
# Preprocessing Resource Initialization
# ============================================================

_STOP_WORDS = None
_LEMMATIZER = None


def initialize_preprocessing():
    """Initialize reusable NLTK preprocessing resources."""

    global _STOP_WORDS, _LEMMATIZER

    if _STOP_WORDS is None or _LEMMATIZER is None:

        download_nltk_resources()

        try:
            _STOP_WORDS = set(stopwords.words("english"))
            _LEMMATIZER = WordNetLemmatizer()

        except LookupError as error:
            raise RuntimeError(
                "Unable to initialize NLTK preprocessing resources."
            ) from error


# ============================================================
# Text Preprocessing
# ============================================================

def preprocess_text(text):
    """
    Clean and preprocess profile text.

    Preserves useful technical terms such as:
    C++, C#, Node.js, Python3 and .NET.
    """

    initialize_preprocessing()

    if pd.isna(text):
        return ""

    text = str(text).lower().strip()

    if not text:
        return ""

    # Preserve selected technical expressions.
    technical_replacements = {
        r"c\+\+": "cplusplus",
        r"c#": "csharp",
        r"\.net\b": "dotnet",
        r"node\.js": "nodejs",
        r"react\.js": "reactjs",
        r"vue\.js": "vuejs",
        r"python\s*3": "python3",
    }

    for pattern, replacement in technical_replacements.items():
        text = re.sub(pattern, replacement, text)

    # Keep letters, numbers and whitespace.
    # This retains technical terms and numerical experience.
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Normalize whitespace.
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()

    # Remove English stopwords.
    tokens = [
        token
        for token in tokens
        if token not in _STOP_WORDS
    ]

    # Lemmatize tokens.
    tokens = [
        _LEMMATIZER.lemmatize(token)
        for token in tokens
    ]

    return " ".join(tokens)


# ============================================================
# Build Profile Text
# ============================================================

def build_profile_text(users_df):
    """Combine professional summary and about-me information."""

    missing_columns = REQUIRED_COLUMNS - set(users_df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required profile columns: "
            f"{sorted(missing_columns)}"
        )

    if users_df.empty:
        raise ValueError(
            "Cannot build profile text from an empty dataset."
        )

    professional_text = (
        users_df["professional_summary"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    about_text = (
        users_df["about_me"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    combined_text = professional_text + " " + about_text

    processed_text = combined_text.apply(preprocess_text)

    empty_profiles = processed_text.str.strip().eq("")

    if empty_profiles.any():
        affected_ids = (
            users_df.loc[empty_profiles, "user_id"]
            .astype(str)
            .tolist()
        )

        raise ValueError(
            "Empty profile text after preprocessing. "
            f"Affected user IDs: {affected_ids}"
        )

    return processed_text


# ============================================================
# Load Embedding Model Name
# ============================================================

def load_embedding_model_name(models_dir):
    """Load the configured embedding model name."""

    model_name_path = (
        Path(models_dir) / "embedding_model_name.joblib"
    )

    if not model_name_path.is_file():
        warnings.warn(
            "Embedding model name artifact not found. "
            f"Using default model: {DEFAULT_MODEL_NAME}",
            stacklevel=2,
        )

        return DEFAULT_MODEL_NAME

    try:
        model_name = joblib.load(model_name_path)

    except Exception as error:
        raise RuntimeError(
            f"Unable to load embedding model configuration: {error}"
        ) from error

    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError(
            "Invalid embedding model name configuration."
        )

    return model_name.strip()


# ============================================================
# Generate NLP Artifacts
# ============================================================

def generate_nlp_artifacts(project_root):
    """
    Generate profile embeddings and cosine similarity matrix.

    Artifacts:
    - models/profile_embeddings.npy
    - models/text_similarity_matrix.npy
    """

    project_root = Path(project_root)

    data_dir = project_root / "data"
    models_dir = project_root / "models"

    users_path = data_dir / "users.csv"

    if not users_path.is_file():
        raise FileNotFoundError(
            f"User dataset not found: {users_path}"
        )

    models_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Load profile dataset
    # --------------------------------------------------------

    try:
        users_df = pd.read_csv(users_path)

    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as error:
        raise RuntimeError(
            f"Unable to load user dataset: {error}"
        ) from error

    if users_df.empty:
        raise ValueError("User dataset is empty.")

    # --------------------------------------------------------
    # Generate preprocessed profile text
    # --------------------------------------------------------

    profile_text = build_profile_text(users_df)

    print("\nNLP PREPROCESSING")
    print("=" * 60)

    print(f"Profiles processed: {len(profile_text)}")
    print("Stopword removal: ENABLED")
    print("Lemmatization: ENABLED")
    print("Technical keyword preservation: ENABLED")

    # --------------------------------------------------------
    # Load Sentence Transformer
    # --------------------------------------------------------

    model_name = load_embedding_model_name(models_dir)

    print("\nSENTENCE TRANSFORMER")
    print("=" * 60)

    print(f"Model: {model_name}")
    print("Loading embedding model...")

    try:
        model = SentenceTransformer(model_name)

    except Exception as error:
        raise RuntimeError(
            f"Unable to load Sentence Transformer '{model_name}'. "
            "Check your internet connection or local model path."
        ) from error

    # --------------------------------------------------------
    # Generate embeddings
    # --------------------------------------------------------

    profile_embeddings = model.encode(
        profile_text.tolist(),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    profile_embeddings = np.asarray(
        profile_embeddings,
        dtype=np.float32,
    )

    if profile_embeddings.ndim != 2:
        raise ValueError(
            "Generated embeddings must be a two-dimensional array."
        )

    if profile_embeddings.shape[0] != len(users_df):
        raise ValueError(
            "Embedding count does not match the number of profiles."
        )

    if profile_embeddings.shape[1] != EXPECTED_EMBEDDING_DIMENSION:
        raise ValueError(
            "Unexpected embedding dimension. "
            f"Expected {EXPECTED_EMBEDDING_DIMENSION}, "
            f"received {profile_embeddings.shape[1]}."
        )

    if not np.isfinite(profile_embeddings).all():
        raise ValueError(
            "Generated embeddings contain invalid numeric values."
        )

    # --------------------------------------------------------
    # Generate cosine similarity matrix
    # --------------------------------------------------------

    text_similarity_matrix = cosine_similarity(
        profile_embeddings
    ).astype(np.float32)

    if text_similarity_matrix.shape != (
        len(users_df),
        len(users_df),
    ):
        raise ValueError(
            "Unexpected similarity matrix dimensions."
        )

    if not np.isfinite(text_similarity_matrix).all():
        raise ValueError(
            "Similarity matrix contains invalid numeric values."
        )

    # --------------------------------------------------------
    # Save artifacts
    # --------------------------------------------------------

    embeddings_path = models_dir / "profile_embeddings.npy"
    similarity_path = models_dir / "text_similarity_matrix.npy"

    np.save(
        embeddings_path,
        profile_embeddings,
    )

    np.save(
        similarity_path,
        text_similarity_matrix,
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print("\nNLP PIPELINE COMPLETED")
    print("=" * 60)

    print(f"Profiles: {len(users_df)}")
    print(
        f"Embedding shape: {profile_embeddings.shape}"
    )
    print(
        f"Similarity matrix shape: {text_similarity_matrix.shape}"
    )

    print("\nSaved artifacts:")
    print(f"- {embeddings_path.relative_to(project_root)}")
    print(f"- {similarity_path.relative_to(project_root)}")

    return profile_embeddings, text_similarity_matrix


# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    generate_nlp_artifacts(PROJECT_ROOT)
