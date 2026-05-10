import numpy as np
from sklearn.decomposition import TruncatedSVD

from .logging_config import get_logger


logger = get_logger(__name__)


class CollaborativeFilteringModel:
    """
    Implements Collaborative Filtering using Matrix Factorization (SVD).

    Why SVD:
    - Reduces dimensionality of user-item interaction matrix
    - Captures latent factors representing user preferences
    - Efficient for sparse data
    - Better generalization to reduce overfitting
    """

    def __init__(self, n_factors: int = 50, random_state: int = 42):
        self.n_factors = n_factors
        self.random_state = random_state
        self.svd = TruncatedSVD(
            n_components=n_factors,
            n_iter=100,
            random_state=random_state
        )
        self.user_factors = None
        self.item_factors = None
        self.movie_ids = None
        self.user_ids = None
        self.user_id_to_index = None
        logger.info(f"CollaborativeFilteringModel initialized with {n_factors} factors")

    def create_interaction_matrix(self, ratings_df):
        """Create user-item interaction matrix from ratings"""
        try:
            logger.info("Creating interaction matrix...")

            self.movie_ids = sorted(ratings_df['movieId'].unique())

            # Create pivot table
            interaction_matrix = ratings_df.pivot_table(
                index='userId',
                columns='movieId',
                values='rating',
                fill_value=0
            )

            self.user_ids = interaction_matrix.index.tolist()
            self.user_id_to_index = {user_id: idx for idx, user_id in enumerate(self.user_ids)}

            logger.info(f"Interaction matrix shape: {interaction_matrix.shape}")
            return interaction_matrix

        except Exception as e:
            logger.error(f"Error creating interaction matrix: {str(e)}")
            return None

    def fit(self, ratings_df):
        """Fit SVD model on interaction matrix"""
        try:
            logger.info("Fitting CollaborativeFiltering model...")

            interaction_matrix = self.create_interaction_matrix(ratings_df)

            # Apply SVD
            self.svd.fit(interaction_matrix)

            # Get factors
            self.user_factors = self.svd.transform(interaction_matrix)  # (n_users, n_factors)
            self.item_factors = self.svd.components_.T  # (n_items, n_factors)

            logger.info(f"User factors shape: {self.user_factors.shape}")
            logger.info(f"Item factors shape: {self.item_factors.shape}")
            logger.info(f"Explained variance ratio: {self.svd.explained_variance_ratio_.sum():.4f}")

            return True

        except Exception as e:
            logger.error(f"Error fitting model: {str(e)}")
            return False

    def predict_score(self, user_id: int, movie_id: int) -> float:
        """Predict rating for user-movie pair"""
        try:
            if movie_id not in self.movie_ids:
                return 0

            movie_idx = self.movie_ids.index(movie_id)

            user_index = self.user_id_to_index.get(user_id) if self.user_id_to_index else None

            # Find user factor (handle new users)
            if user_index is not None:
                user_factor = self.user_factors[user_index]
            else:
                user_factor = np.zeros(self.n_factors)

            # Compute dot product
            score = np.dot(user_factor, self.item_factors[movie_idx])
            return float(np.clip(score, 0, 5))

        except Exception as e:
            logger.error(f"Error in predict_score: {str(e)}")
            return 0
