from typing import List, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .logging_config import get_logger


logger = get_logger(__name__)


class ContentBasedModel:
    """
    Implements Content-Based Filtering using cosine similarity on movie features.

    Why cosine similarity:
    - Measures angle between feature vectors (0-1 scale)
    - Robust to magnitude differences
    - Efficient for sparse vectors
    - Works well with tag-based features
    """

    def __init__(self):
        self.movie_features = None
        self.movie_ids = None
        self.similarity_matrix = None
        logger.info("ContentBasedModel initialized")

    def fit(self, movie_ids, movie_features):
        """
        Fit model by computing cosine similarity between all movies

        Args:
            movie_ids: List of movie IDs
            movie_features: Feature matrix of shape (n_movies, n_features)
        """
        try:
            logger.info("Fitting ContentBased model...")

            self.movie_ids = movie_ids.tolist()
            self.movie_features = movie_features

            # Compute cosine similarity matrix
            self.similarity_matrix = cosine_similarity(movie_features)

            logger.info(f"Similarity matrix shape: {self.similarity_matrix.shape}")
            logger.info(
                "Similarity stats - Mean: %.4f, Max: %.4f, Min: %.4f",
                self.similarity_matrix.mean(),
                self.similarity_matrix.max(),
                self.similarity_matrix.min()
            )

            return True

        except Exception as e:
            logger.error(f"Error fitting model: {str(e)}")
            return False

    def find_similar_movies(self, movie_id: int, top_k: int = 10) -> List[Tuple[int, float]]:
        """Find top-k similar movies based on content features"""
        try:
            if movie_id not in self.movie_ids:
                return []

            movie_idx = self.movie_ids.index(movie_id)
            similarities = self.similarity_matrix[movie_idx]

            # Get top-k indices (excluding the movie itself)
            top_indices = np.argsort(similarities)[::-1][1:top_k + 1]

            results = [
                (self.movie_ids[idx], float(similarities[idx]))
                for idx in top_indices
            ]

            return results

        except Exception as e:
            logger.error(f"Error in find_similar_movies: {str(e)}")
            return []
