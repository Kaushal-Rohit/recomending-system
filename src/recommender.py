"""
Recommendation engine.

Provides two complementary algorithms:

1. ContentBasedRecommender
   - Builds TF-IDF vectors from item metadata (genre, tags, creator, etc.)
   - Creates a per-user profile vector as the weighted average of interacted items
   - Returns items ranked by cosine similarity to the user profile

2. CollaborativeRecommender
   - Uses the user-item interaction-score matrix
   - Finds the K most similar users via cosine similarity
   - Scores unseen items as the similarity-weighted average score from those neighbours

Both classes expose the same interface:  recommend(user_id, n=10) → list[dict]
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from src.data_loader import DataLoader
from src.data_models import ContentItem

logger = logging.getLogger(__name__)


class ContentBasedRecommender:
    """
    Content-based recommendation using TF-IDF on item metadata.
    """

    def __init__(self, loader: DataLoader):
        self._loader = loader
        self._tfidf_matrix: Optional[np.ndarray] = None
        self._item_ids: list[str] = []
        self._items: dict[str, ContentItem] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(self) -> "ContentBasedRecommender":
        """Build the TF-IDF matrix from the content catalogue."""
        from sklearn.feature_extraction.text import TfidfVectorizer

        items = self._loader.get_catalog_items()
        if not items:
            raise ValueError("Content catalogue is empty — cannot fit the recommender.")

        self._items = {item.item_id: item for item in items}
        self._item_ids = [item.item_id for item in items]
        feature_strings = [item.feature_string for item in items]

        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            stop_words="english",
        )
        self._tfidf_matrix = vectorizer.fit_transform(feature_strings).toarray()
        logger.info("ContentBasedRecommender fitted on %d items.", len(items))
        return self

    def recommend(self, user_id: str, n: int = 10) -> list[dict]:
        """Return the top-N content-based recommendations for *user_id*."""
        if self._tfidf_matrix is None:
            self.fit()

        user_profile = self._build_user_profile(user_id)
        if user_profile is None:
            logger.warning("No history found for user '%s'; returning popular items.", user_id)
            return self._popular_items(n)

        # Cosine similarity between user profile and every item
        scores = cosine_similarity(user_profile.reshape(1, -1), self._tfidf_matrix)[0]

        # Exclude items the user has already interacted with
        seen_items = self._seen_items(user_id)
        results = []
        for idx in np.argsort(scores)[::-1]:
            item_id = self._item_ids[idx]
            if item_id in seen_items:
                continue
            item = self._items[item_id]
            results.append({
                "item_id": item_id,
                "title": item.title,
                "content_type": item.content_type.value if hasattr(item.content_type, "value") else item.content_type,
                "genre": item.genre,
                "creator": item.creator,
                "platform": item.platform,
                "score": round(float(scores[idx]), 4),
            })
            if len(results) >= n:
                break

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_user_profile(self, user_id: str) -> Optional[np.ndarray]:
        """Weighted average of TF-IDF vectors for items the user interacted with."""
        matrix = self._loader.get_user_item_matrix()
        if user_id not in matrix.index:
            return None

        user_scores = matrix.loc[user_id]
        positive_items = user_scores[user_scores > 0]
        if positive_items.empty:
            return None

        # Only include items that are in the catalogue
        common_items = [iid for iid in positive_items.index if iid in self._items]
        if not common_items:
            return None

        weights = positive_items[common_items].values.astype(float)
        item_indices = [self._item_ids.index(iid) for iid in common_items]
        vectors = self._tfidf_matrix[item_indices]

        # Weighted sum, then normalise
        profile = np.average(vectors, axis=0, weights=weights)
        norm = np.linalg.norm(profile)
        return profile / norm if norm > 0 else profile

    def _seen_items(self, user_id: str) -> set[str]:
        matrix = self._loader.get_user_item_matrix()
        if user_id not in matrix.index:
            return set()
        return set(matrix.columns[matrix.loc[user_id] != 0].tolist())

    def _popular_items(self, n: int) -> list[dict]:
        """Fallback: return the first N items in the catalogue."""
        items = self._loader.get_catalog_items()[:n]
        return [
            {
                "item_id": item.item_id,
                "title": item.title,
                "content_type": item.content_type.value if hasattr(item.content_type, "value") else item.content_type,
                "genre": item.genre,
                "creator": item.creator,
                "platform": item.platform,
                "score": 0.0,
            }
            for item in items
        ]


class CollaborativeRecommender:
    """
    User-based collaborative filtering using cosine similarity on the
    user-item interaction matrix.
    """

    def __init__(self, loader: DataLoader, n_neighbors: int = 10):
        self._loader = loader
        self._n_neighbors = n_neighbors

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def recommend(self, user_id: str, n: int = 10) -> list[dict]:
        """Return the top-N collaborative recommendations for *user_id*."""
        matrix = self._loader.get_user_item_matrix()

        if user_id not in matrix.index:
            logger.warning("User '%s' not in matrix; returning popular items.", user_id)
            return self._popular_items(n)

        # Normalise rows to unit length (cosine sim equivalent)
        normed = normalize(matrix.values, norm="l2")
        user_index = list(matrix.index).index(user_id)
        user_vec = normed[user_index].reshape(1, -1)

        # Similarity to all other users
        sims = cosine_similarity(user_vec, normed)[0]
        sims[user_index] = -1.0  # exclude self

        # Top-K neighbour indices
        k = min(self._n_neighbors, len(sims) - 1)
        top_k_indices = np.argsort(sims)[::-1][:k]
        top_k_sims = sims[top_k_indices]

        if top_k_sims.sum() == 0:
            return self._popular_items(n)

        # Score each item as the sim-weighted average of neighbour scores
        neighbour_scores = matrix.iloc[top_k_indices]  # (k × items)
        weights = top_k_sims.reshape(-1, 1)             # (k × 1)
        weighted_sum = (neighbour_scores.values * weights).sum(axis=0)
        total_weight = np.abs(weights).sum()
        item_scores = weighted_sum / total_weight if total_weight > 0 else weighted_sum

        # Exclude items the user already interacted with
        user_row = matrix.loc[user_id]
        seen_items = set(matrix.columns[user_row != 0].tolist())

        all_items_in_catalog = {item.item_id: item for item in self._loader.get_catalog_items()}
        item_ids = list(matrix.columns)

        results = []
        for idx in np.argsort(item_scores)[::-1]:
            item_id = item_ids[idx]
            if item_id in seen_items:
                continue
            if item_id not in all_items_in_catalog:
                continue
            item = all_items_in_catalog[item_id]
            results.append({
                "item_id": item_id,
                "title": item.title,
                "content_type": item.content_type.value if hasattr(item.content_type, "value") else item.content_type,
                "genre": item.genre,
                "creator": item.creator,
                "platform": item.platform,
                "score": round(float(item_scores[idx]), 4),
            })
            if len(results) >= n:
                break

        return results if results else self._popular_items(n)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _popular_items(self, n: int) -> list[dict]:
        """Fallback: items with the highest total interaction score."""
        matrix = self._loader.get_user_item_matrix()
        if matrix.empty:
            items = self._loader.get_catalog_items()[:n]
            return [
                {
                    "item_id": item.item_id,
                    "title": item.title,
                    "content_type": item.content_type.value if hasattr(item.content_type, "value") else item.content_type,
                    "genre": item.genre,
                    "creator": item.creator,
                    "platform": item.platform,
                    "score": 0.0,
                }
                for item in items
            ]

        totals = matrix.sum(axis=0).sort_values(ascending=False)
        all_items = {item.item_id: item for item in self._loader.get_catalog_items()}
        results = []
        for item_id, score in totals.head(n).items():
            if item_id in all_items:
                item = all_items[item_id]
                results.append({
                    "item_id": item_id,
                    "title": item.title,
                    "content_type": item.content_type.value if hasattr(item.content_type, "value") else item.content_type,
                    "genre": item.genre,
                    "creator": item.creator,
                    "platform": item.platform,
                    "score": round(float(score), 4),
                })
        return results
