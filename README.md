# Intelligent Hybrid Recommendation System

## 🚀 Live Demo

[Launch Intelligent Hybrid Recommendation System](https://intelligent-hybrid-recommendation-system-8uh9zqy9pjcy89jvlsthj.streamlit.app/)

AI-Powered Profile Matching Using NLP, Semantic Similarity and Adaptive Learning

## Project Overview

The **Intelligent Hybrid Recommendation System** is an AI-powered application designed to identify and recommend compatible user profiles by combining Natural Language Processing (NLP), semantic similarity, personality compatibility, and location-based matching.

The system analyzes users' professional summaries, personal descriptions, personality types, and locations to generate personalized recommendations.

It combines multiple matching techniques into a unified scoring framework and provides an interactive interface for exploring recommendations, understanding match explanations, and collecting user feedback.

The project is developed using Python and machine learning technologies, with a modular architecture and an interactive Streamlit-based application.

## Key Features

- **Profile-Based Matching:** Matches users using their professional and personal profile information.
- **NLP-Based Text Processing:** Cleans and preprocesses profile descriptions using stopword removal and lemmatization.
- **Semantic Similarity:** Uses Sentence Transformers to generate meaningful profile embeddings.
- **Cosine Similarity:** Measures semantic similarity between user profiles.
- **MBTI Compatibility:** Incorporates personality compatibility using a predefined rule-based compatibility matrix.
- **Location-Based Matching:** Considers geographical location as a matching factor.
- **Hybrid Scoring:** Combines text similarity, MBTI compatibility, and location compatibility using weighted scoring.
- **Adaptive Feedback:** Collects accept/reject feedback and supports personalized recommendation weighting.
- **Match Explanations:** Displays individual compatibility components to help users understand recommendations.
- **Top-5 Recommendations:** Generates the five highest-scoring matches for a selected profile.
- **Interactive Dashboard:** Provides a Streamlit interface for profile matching, feedback collection, and recommendation exploration.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| NLP | NLTK |
| Semantic Embeddings | Sentence Transformers |
| Similarity Measurement | Cosine Similarity |
| Visualization | Matplotlib |
| Model Storage | Joblib, NumPy |
| Frontend / Dashboard | Streamlit |
| Development Environment | VS Code, Jupyter Notebook |
| Version Control | Git, GitHub |

## Hybrid Matching Methodology

The system evaluates profile compatibility using three major components:

1. **Text Similarity:** Measures semantic similarity between professional summaries and personal descriptions using Sentence Transformer embeddings.
2. **MBTI Compatibility:** Evaluates personality compatibility using a predefined 16-type compatibility matrix.
3. **Location Compatibility:** Assigns compatibility scores based on whether users share the same location.

The final compatibility score is calculated using a weighted scoring approach:

\[
S = (w_1 \times T) + (w_2 \times M) + (w_3 \times L)
\]

Where:

- \(S\) = Final compatibility score
- \(T\) = Text similarity score
- \(M\) = MBTI compatibility score
- \(L\) = Location compatibility score
- \(w_1,w_2,w_3\) = Component weights

Initial weights:

- Text Similarity: 0.50
- MBTI Compatibility: 0.30
- Location Compatibility: 0.20

The weights can be adjusted through the adaptive feedback workflow.

## Project Architecture

The project follows a modular architecture that separates data processing, NLP, recommendation logic, adaptive learning, and application functionality.

```text
intelligent-hybrid-recommendation-system/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── users.csv
│   └── feedback.csv
│
├── models/
│   ├── embedding_model_name.joblib
│   ├── initial_weights.joblib
│   ├── mbti_compatibility.joblib
│   ├── personalized_weights.joblib
│   ├── profile_embeddings.npy
│   ├── text_similarity_matrix.npy
│   ├── notebook_experimental_adaptive_models.joblib
│   └── notebook_experimental_personalized_weights.joblib
│
├── notebooks/
│   └── intelligent_hybrid_recommendation_system.ipynb
│
├── outputs/
│   ├── feedback_over_time.csv
│   ├── performance_results.csv
│   ├── project_summary.csv
│   └── score_comparison.csv
│
├── src/
│   ├── adaptive_updater.py
│   ├── data_pipeline.py
│   ├── matcher.py
│   ├── nlp_pipeline.py
│   └── performance_analysis.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

### Model Artifacts

- **Production artifacts:** Store the embedding model identifier, initial scoring weights, MBTI compatibility matrix, personalized weights, profile embeddings, and similarity matrix used by the application.
- **Experimental artifacts:** Store the Logistic Regression models and experimental personalized weights generated by the research notebook.

*Note: The architecture above represents the intended project structure. Verify that each listed file exists in the final project directory before publishing to GitHub. Experimental artifacts are optional for running the Streamlit application.*

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Ariyan-123/intelligent-hybrid-recommendation-system.git
```

### 2. Navigate to the Project Directory

```bash
cd intelligent-hybrid-recommendation-system
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 4. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Run the Streamlit Application

```bash
python -m streamlit run app/streamlit_app.py
```

The application will open in your browser.

## Application Modules

| Module | Description |
|---|---|
| Data Pipeline | Cleans and validates user and feedback datasets. |
| NLP Pipeline | Processes profile descriptions and generates semantic embeddings. |
| Hybrid Matcher | Calculates compatibility scores and generates recommendations. |
| Adaptive Updater | Updates personalized recommendation weights based on feedback. |
| Performance Analysis | Examines feedback patterns and recommendation score distributions. |
| Streamlit Application | Provides the interactive user interface. |

## Dataset Information

The project uses synthetically generated user profiles and simulated feedback interactions for development and demonstration.

The profile dataset includes:

- User ID and name
- Age and location
- Profession and experience
- Professional summary
- Personal description
- MBTI personality type
- Interests

The feedback dataset contains:

- User ID
- Matched user ID
- Accept/reject action
- Interaction timestamp

**Dataset Disclaimer:** The generated profiles and feedback records are synthetic. They do not represent actual users or verified real-world preferences.

## Adaptive Learning and Personalization

The application supports personalized recommendation weighting through an adaptive feedback mechanism.

When users accept or reject recommendations, the system records their feedback and updates their personalized matching weights.

The experimental notebook also explores Logistic Regression-based learning to analyze the influence of text similarity, MBTI compatibility, and location features.

**Note:** The Logistic Regression experiments use simulated feedback and are separate from the production adaptive updater. Their results demonstrate the learning workflow and do not establish improved real-world recommendation accuracy.

## Important Limitations

- The MBTI compatibility matrix is predefined and heuristic; it is not scientifically validated.
- The initial dataset contains synthetic profiles and simulated feedback.
- Experimental per-user learning models have limited feedback data.
- Recommendation quality requires evaluation on independent, real-world or appropriately held-out data.
- Compatibility scores represent algorithmic matching estimates, not guaranteed personal or professional compatibility.

## Future Enhancements

- Integration of real-world profile datasets with appropriate consent.
- Improved recommendation models using larger feedback datasets.
- Chronological training and independent testing for adaptive learning.
- More advanced NLP and contextual profile understanding.
- Improved recommendation explanations.
- Scalable database integration.
- User authentication and profile management.
- Cloud deployment and production monitoring.

## Project Objective

The objective of this project is to demonstrate how NLP, machine learning, semantic similarity, personality-based compatibility, and adaptive feedback can be integrated into a unified recommendation system.

The application provides a practical demonstration of intelligent profile matching through a modular Python architecture and an interactive Streamlit interface.

---

**Project:** Intelligent Hybrid Recommendation System  
**Category:** Artificial Intelligence / Machine Learning / NLP  
**Interface:** Streamlit