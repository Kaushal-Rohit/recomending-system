# Hybrid Recommendation System Report

## 1. Purpose
This project implements a hybrid movie recommendation engine that combines collaborative filtering, content-based filtering, and user-behavior learning. The goal is to provide personalized, robust recommendations while handling cold-start cases and improving relevance.

## 2. Codebase Split (Reusable Parts)
The original single-file implementation has been split into focused modules to improve reuse, testing, and maintenance.

| Module | Role |
| --- | --- |
| `recommender/logging_config.py` | Centralized logging setup shared by all modules. |
| `recommender/data_preprocessor.py` | Loads CSV datasets, merges them, and builds movie feature vectors. |
| `recommender/collaborative_filtering.py` | Matrix factorization (SVD) for user-item interaction modeling. |
| `recommender/content_based.py` | Cosine similarity engine for content-based recommendations. |
| `recommender/user_behavior.py` | Gradient boosting model for user preference prediction. |
| `recommender/hybrid.py` | Orchestrates the full pipeline and combines model scores. |
| `recommendation_model.py` | CLI entrypoint and backwards-compatible exports. |

## 3. Data Pipeline
1. **Load CSVs**: `movies.csv`, `tags.csv`, `ratings.csv`, `genome-scores.csv`.
2. **Merge datasets**:
   - Tags are aggregated per movie.
   - Ratings are joined with movie metadata and genome relevance.
3. **Feature creation**:
   - Tags are binarized into a sparse feature vector.
   - Genome relevance score is normalized and appended.

## 4. Model Components (Detailed)

### 4.1 Collaborative Filtering (SVD)
- **Input**: User–movie rating matrix.
- **Method**: Truncated SVD decomposes the interaction matrix into latent factors.
- **Output**: Predicted rating from user and item latent vectors.
- **Strengths**: Learns latent preferences and improves with more ratings.

### 4.2 Content-Based Filtering
- **Input**: Movie feature vectors (tag + genome relevance).
- **Method**: Cosine similarity between movie vectors.
- **Output**: Similarity score between movies, scaled to a 0–5 range.
- **Strengths**: Works for cold-start items with known content features.

### 4.3 User Behavior Model
- **Input features**:
  - User rating mean, std, count
  - Movie rating mean, count
  - Genome relevance
  - Rating deviation and interaction terms
- **Model**: GradientBoostingRegressor with feature scaling.
- **Output**: Predicted user preference score (0–5).
- **Strengths**: Captures non-linear user behavior trends.

### 4.4 Hybrid Scoring
The final recommendation score is a weighted sum:
- **40%** Collaborative Filtering
- **35%** Content-Based
- **25%** User Behavior

This blend balances personalization, relevance, and cold-start handling.

## 5. Recommendation Flow
1. Identify movies the user has already rated.
2. For each unrated movie:
   - Compute collaborative filtering score.
   - Compute content-based similarity score.
   - Compute user-behavior prediction.
3. Combine scores using the hybrid weights.
4. Rank and return the top results.

## 6. Performance and Evaluation
During training, the system logs key performance indicators:

### Logged Metrics
- **Collaborative Filtering**: SVD explained variance ratio (how much of the rating variance is captured).
- **Content-Based**: Similarity matrix statistics (mean, min, max).
- **User Behavior**: R² score on train and test splits.

### How to Capture Metrics
Run the entrypoint and review `recommendation_model.log`:
```bash
python recommendation_model.py
```
The log records the metrics above for the current dataset. These metrics provide:
- **Generalization check** (train vs. test R²).
- **Latent factor quality** (SVD explained variance).
- **Content similarity distribution** (cosine similarity stats).

### Interpreting Results
- Higher **R²** indicates better predictive power for user behavior.
- Higher **explained variance** indicates stronger collaborative factors.
- Reasonable **similarity ranges** avoid over-clustering similar titles.

## 7. How to Use
Run training and generate recommendations:
```bash
python recommendation_model.py
```
The script saves `hybrid_recommender.pkl` and prints example recommendations.
