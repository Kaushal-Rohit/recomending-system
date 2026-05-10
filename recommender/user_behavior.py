import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .logging_config import get_logger


logger = get_logger(__name__)


class UserBehaviorModel:
    """
    Implements user behavior learning using gradient boosting.

    Predicts user preference score based on:
    - User's rating history (average, std, count)
    - Movie's average rating
    - Relevance score
    - Temporal patterns (if timestamp available)

    Why Gradient Boosting:
    - Handles non-linear relationships
    - Feature importance analysis
    - Better generalization with early stopping
    - Robust to outliers
    """

    def __init__(self, random_state: int = 42):
        self.model = GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=6,
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.8,
            random_state=random_state,
            validation_fraction=0.1,
            n_iter_no_change=20
        )
        self.scaler = StandardScaler()
        self.feature_names = None
        self.feature_importance = None
        logger.info("UserBehaviorModel initialized")

    def engineer_features(self, merged_data):
        """
        Engineer features for user behavior prediction.

        Features created:
        1. user_rating_mean: Average rating given by user
        2. user_rating_std: Standard deviation of user's ratings
        3. user_num_ratings: Count of ratings given by user
        4. movie_avg_rating: Average rating received by movie
        5. movie_num_ratings: Count of ratings for movie
        6. relevance_score: Genome relevance score
        7. user_novelty_score: How new movies are compared to user's watch history
        """
        try:
            logger.info("Engineering features for user behavior...")

            # Create copy to avoid warnings
            data = merged_data.copy()

            # User-level aggregations
            user_stats = data.groupby('userId').agg({
                'rating': ['mean', 'std', 'count']
            }).reset_index()
            user_stats.columns = ['userId', 'user_rating_mean', 'user_rating_std', 'user_num_ratings']
            user_stats['user_rating_std'].fillna(0, inplace=True)

            # Movie-level aggregations
            movie_stats = data.groupby('movieId').agg({
                'rating': ['mean', 'count']
            }).reset_index()
            movie_stats.columns = ['movieId', 'movie_avg_rating', 'movie_num_ratings']

            # Merge features
            features = data.merge(user_stats, on='userId', how='left')
            features = features.merge(movie_stats, on='movieId', how='left')

            # Handle missing values
            features['user_rating_std'].fillna(0, inplace=True)
            features['movie_avg_rating'].fillna(features['rating'].mean(), inplace=True)
            features['movie_num_ratings'].fillna(1, inplace=True)
            features['avg_relevance_score'].fillna(0.5, inplace=True)

            # Additional features
            features['rating_diff'] = features['rating'] - features['movie_avg_rating']
            features['user_movie_interaction'] = features['user_rating_mean'] * features['movie_avg_rating']

            # Normalize user ratings (0-1 scale)
            features['user_rating_mean_norm'] = features['user_rating_mean'] / 5.0

            self.feature_names = [
                'user_rating_mean', 'user_rating_std', 'user_num_ratings',
                'movie_avg_rating', 'movie_num_ratings', 'avg_relevance_score',
                'rating_diff', 'user_movie_interaction', 'user_rating_mean_norm'
            ]

            logger.info(f"Created {len(self.feature_names)} features: {self.feature_names}")

            return features

        except Exception as e:
            logger.error(f"Error in feature engineering: {str(e)}")
            return None

    def fit(self, features_data, target='rating'):
        """Fit user behavior model"""
        try:
            logger.info("Fitting UserBehavior model...")

            X = features_data[self.feature_names].values
            y = features_data[target].values

            # Standardize features
            X_scaled = self.scaler.fit_transform(X)

            # Train-test split (80-20)
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )

            logger.info(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")

            # Fit model
            self.model.fit(X_train, y_train)

            # Evaluate
            train_score = self.model.score(X_train, y_train)
            test_score = self.model.score(X_test, y_test)

            logger.info(f"Training R² score: {train_score:.4f}")
            logger.info(f"Testing R² score: {test_score:.4f}")
            logger.info(f"Overfitting check - Difference: {abs(train_score - test_score):.4f}")

            # Feature importance
            self.feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)

            logger.info("Feature Importance:")
            logger.info(self.feature_importance.to_string())

            return True

        except Exception as e:
            logger.error(f"Error fitting model: {str(e)}")
            return False

    def predict(self, features_data):
        """Predict user preference scores"""
        try:
            X = features_data[self.feature_names].values
            X_scaled = self.scaler.transform(X)
            predictions = self.model.predict(X_scaled)
            return np.clip(predictions, 0, 5)

        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            return np.zeros(len(features_data))
