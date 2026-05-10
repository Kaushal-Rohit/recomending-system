import pickle
from typing import Dict, List

import numpy as np

from .collaborative_filtering import CollaborativeFilteringModel
from .content_based import ContentBasedModel
from .data_preprocessor import DataPreprocessor
from .logging_config import get_logger
from .user_behavior import UserBehaviorModel


logger = get_logger(__name__)


class HybridRecommendationSystem:
    """
    Hybrid Recommendation System combining:
    1. Collaborative Filtering (40% weight)
    2. Content-Based Filtering (35% weight)
    3. User Behavior Learning (25% weight)

    This hybrid approach provides:
    - Cold-start handling (content-based for new users/movies)
    - Serendipity (collaborative filtering discovers new preferences)
    - Personalization (user behavior learning)
    - Robustness (ensemble reduces individual model errors)
    """

    def __init__(self):
        self.cf_model = CollaborativeFilteringModel(n_factors=50)
        self.cb_model = ContentBasedModel()
        self.ub_model = UserBehaviorModel()

        self.preprocessor = DataPreprocessor()
        self.unique_movies = None
        self.movie_features = None
        self.tags_list = None
        self.merged_data = None

        # Weights for hybrid approach
        self.cf_weight = 0.40
        self.cb_weight = 0.35
        self.ub_weight = 0.25

        logger.info("HybridRecommendationSystem initialized")

    def train(self, data_dir: str = '.'):
        """Complete training pipeline"""
        try:
            logger.info("=" * 60)
            logger.info("Starting HybridRecommendationSystem Training")
            logger.info("=" * 60)

            # Step 1: Load and preprocess data
            logger.info("\n[Step 1] Loading and preprocessing data...")
            self.preprocessor = DataPreprocessor(data_dir=data_dir)
            self.preprocessor.load_data()
            self.merged_data = self.preprocessor.merge_data()

            if self.merged_data is None:
                logger.error("Failed to merge data")
                return False

            # Step 2: Create movie features
            logger.info("\n[Step 2] Creating movie features...")
            self.unique_movies, self.movie_features, self.tags_list = (
                self.preprocessor.create_movie_features()
            )

            if self.movie_features is None:
                logger.error("Failed to create movie features")
                return False

            # Step 3: Train Collaborative Filtering
            logger.info("\n[Step 3] Training Collaborative Filtering model...")
            self.cf_model.fit(self.merged_data)

            # Step 4: Train Content-Based Model
            logger.info("\n[Step 4] Training Content-Based model...")
            self.cb_model.fit(
                self.unique_movies['movieId'].values,
                self.movie_features
            )

            # Step 5: Train User Behavior Model
            logger.info("\n[Step 5] Training User Behavior model...")
            features_data = self.ub_model.engineer_features(self.merged_data)
            self.ub_model.fit(features_data)

            logger.info("\n" + "=" * 60)
            logger.info("Training Complete!")
            logger.info("=" * 60)

            return True

        except Exception as e:
            logger.error(f"Error in training: {str(e)}")
            return False

    def recommend(self, user_id: int, num_recommendations: int = 10,
                  current_movie_id: int = None) -> List[Dict]:
        """
        Generate recommendations for a user.

        Strategy:
        1. Get user's watch history and average preferences
        2. For each unrated movie:
           a. Get collaborative filtering score
           b. Get content-based similarity to watch history
           c. Get user behavior prediction score
           d. Combine with weights to get hybrid score
        3. Return top-k recommendations

        Args:
            user_id: User ID
            num_recommendations: Number of recommendations to return
            current_movie_id: Current movie being watched (for content-based similarity)

        Returns:
            List of dicts with movie info and scores
        """
        try:
            logger.info(f"\nGenerating recommendations for user {user_id}...")

            # Get user's watch history
            user_ratings = self.merged_data[self.merged_data['userId'] == user_id]
            watched_movies = set(user_ratings['movieId'].unique())

            logger.info(f"User has watched {len(watched_movies)} movies")
            logger.info(f"User's average rating: {user_ratings['rating'].mean():.2f}")

            # Get all unrated movies
            all_movies = set(self.unique_movies['movieId'].unique())
            unrated_movies = all_movies - watched_movies

            logger.info(f"Generating scores for {len(unrated_movies)} unrated movies...")

            recommendations = []

            for movie_id in unrated_movies:
                try:
                    # 1. Collaborative Filtering Score
                    cf_score = self.cf_model.predict_score(user_id, movie_id)

                    # 2. Content-Based Score
                    cb_score = 0
                    if current_movie_id and current_movie_id in self.cf_model.movie_ids:
                        similar_movies = self.cb_model.find_similar_movies(
                            current_movie_id, top_k=1000
                        )
                        for sim_movie_id, sim_score in similar_movies:
                            if sim_movie_id == movie_id:
                                cb_score = sim_score * 5  # Scale to 0-5
                                break
                    else:
                        # If no current movie, use average similarity to watch history
                        if watched_movies:
                            scores = []
                            for watched_id in list(watched_movies)[:20]:  # Limit for efficiency
                                similar = self.cb_model.find_similar_movies(watched_id, top_k=100)
                                for sim_id, sim_score in similar:
                                    if sim_id == movie_id:
                                        scores.append(sim_score)
                            if scores:
                                cb_score = np.mean(scores) * 5

                    # 3. User Behavior Score
                    ub_features = self.ub_model.engineer_features(
                        self.merged_data[self.merged_data['movieId'] == movie_id]
                    )
                    if len(ub_features) > 0:
                        ub_score = self.ub_model.predict(ub_features)[0]
                    else:
                        ub_score = user_ratings['rating'].mean()

                    # Hybrid score (weighted combination)
                    hybrid_score = (
                        self.cf_weight * cf_score +
                        self.cb_weight * cb_score +
                        self.ub_weight * ub_score
                    )

                    # Get movie info
                    movie_info = self.unique_movies[
                        self.unique_movies['movieId'] == movie_id
                    ]

                    if len(movie_info) > 0:
                        movie_row = movie_info.iloc[0]
                        recommendations.append({
                            'movieId': int(movie_id),
                            'title': movie_row['title'],
                            'genres': movie_row['genres'],
                            'tags': movie_row['tag_list'],
                            'hybrid_score': float(hybrid_score),
                            'cf_score': float(cf_score),
                            'cb_score': float(cb_score),
                            'ub_score': float(ub_score),
                            'avg_relevance_score': float(movie_row['avg_relevance_score'])
                        })

                except Exception as e:
                    logger.debug(f"Error scoring movie {movie_id}: {str(e)}")
                    continue

            # Sort by hybrid score and return top-k
            recommendations.sort(key=lambda x: x['hybrid_score'], reverse=True)
            recommendations = recommendations[:num_recommendations]

            logger.info(f"Generated {len(recommendations)} recommendations")

            return recommendations

        except Exception as e:
            logger.error(f"Error in recommendation: {str(e)}")
            return []

    def save_model(self, filepath: str = 'hybrid_recommender.pkl'):
        """Save trained model to disk"""
        try:
            model_dict = {
                'cf_model': self.cf_model,
                'cb_model': self.cb_model,
                'ub_model': self.ub_model,
                'unique_movies': self.unique_movies,
                'movie_features': self.movie_features,
                'tags_list': self.tags_list,
                'cf_weight': self.cf_weight,
                'cb_weight': self.cb_weight,
                'ub_weight': self.ub_weight
            }

            with open(filepath, 'wb') as f:
                pickle.dump(model_dict, f)

            logger.info(f"Model saved to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False

    def load_model(self, filepath: str = 'hybrid_recommender.pkl'):
        """Load trained model from disk"""
        try:
            with open(filepath, 'rb') as f:
                model_dict = pickle.load(f)

            self.cf_model = model_dict['cf_model']
            self.cb_model = model_dict['cb_model']
            self.ub_model = model_dict['ub_model']
            self.unique_movies = model_dict['unique_movies']
            self.movie_features = model_dict['movie_features']
            self.tags_list = model_dict['tags_list']
            self.cf_weight = model_dict['cf_weight']
            self.cb_weight = model_dict['cb_weight']
            self.ub_weight = model_dict['ub_weight']

            logger.info(f"Model loaded from {filepath}")
            return True

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
