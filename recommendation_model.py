"""
Movie Recommendation System using Hybrid Approach.

This module provides the CLI entrypoint and re-exports the core classes for
backward compatibility.
"""

import warnings

from recommender import (
    CollaborativeFilteringModel,
    ContentBasedModel,
    DataPreprocessor,
    HybridRecommendationSystem,
    UserBehaviorModel,
)
from recommender.logging_config import get_logger


warnings.filterwarnings('ignore')

logger = get_logger(__name__)

__all__ = [
    "CollaborativeFilteringModel",
    "ContentBasedModel",
    "DataPreprocessor",
    "HybridRecommendationSystem",
    "UserBehaviorModel",
]


if __name__ == "__main__":
    # Initialize and train system
    recommender = HybridRecommendationSystem()
    recommender.train()

    # Save trained model
    recommender.save_model('hybrid_recommender.pkl')

    # Example: Get recommendations for a user
    user_id = 1
    recommendations = recommender.recommend(user_id, num_recommendations=10)

    logger.info(f"\nTop 10 recommendations for user {user_id}:")
    for i, rec in enumerate(recommendations, 1):
        logger.info(f"{i}. {rec['title']} (Score: {rec['hybrid_score']:.3f})")
