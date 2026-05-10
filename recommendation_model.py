"""
Movie Recommendation System using Hybrid Approach
This module implements a sophisticated recommendation engine that combines:
1. Collaborative Filtering (user-item interactions)
2. Content-Based Filtering (movie tags and features)
3. Relevance Scoring (genome-scores)
4. User Behavior Learning
"""

import pandas as pd
import numpy as np
import pickle
import warnings
from typing import List, Tuple, Dict
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MultiLabelBinarizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.regularization import l1_l2_penalty
import xgboost as xgb
from scipy.sparse import csr_matrix, vstack
from datetime import datetime
import logging

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('recommendation_model.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Handles data loading, merging, and preprocessing for the recommendation system.
    
    Features:
    - Loads all CSV files
    - Merges data on movie and user IDs
    - Handles missing values
    - Normalizes data distributions
    """
    
    def __init__(self, data_dir: str = '.'):
        self.data_dir = data_dir
        self.movies = None
        self.tags = None
        self.ratings = None
        self.genome_scores = None
        self.merged_data = None
        logger.info("DataPreprocessor initialized")
    
    def load_data(self):
        """Load all CSV files"""
        try:
            logger.info("Loading movies data...")
            self.movies = pd.read_csv(f'{self.data_dir}/movies.csv')
            
            logger.info("Loading tags data...")
            self.tags = pd.read_csv(f'{self.data_dir}/tags.csv')
            
            logger.info("Loading ratings data...")
            self.ratings = pd.read_csv(f'{self.data_dir}/ratings.csv')
            
            logger.info("Loading genome-scores data...")
            self.genome_scores = pd.read_csv(f'{self.data_dir}/genome-scores.csv')
            
            logger.info(f"Movies: {self.movies.shape}, Tags: {self.tags.shape}, "
                       f"Ratings: {self.ratings.shape}, Genome: {self.genome_scores.shape}")
            
            return True
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return False
    
    def merge_data(self):
        """
        Merge all datasets on movieId and userId.
        Creates a unified dataset for training.
        """
        try:
            logger.info("Starting data merge process...")
            
            # Merge movies with tags
            movie_tags = self.tags.groupby('movieId')['tag'].apply(list).reset_index()
            movie_tags.columns = ['movieId', 'tag_list']
            
            movies_with_tags = self.movies.merge(movie_tags, on='movieId', how='left')
            movies_with_tags['tag_list'] = movies_with_tags['tag_list'].fillna('').apply(
                lambda x: [] if x == '' else x
            )
            
            # Merge with ratings
            merged = self.ratings.merge(
                movies_with_tags,
                on='movieId',
                how='left'
            )
            
            # Merge with genome scores
            genome_avg = self.genome_scores.groupby('movieId')['relevanceScore'].mean().reset_index()
            genome_avg.columns = ['movieId', 'avg_relevance_score']
            
            self.merged_data = merged.merge(
                genome_avg,
                on='movieId',
                how='left'
            )
            
            # Fill missing values
            self.merged_data['avg_relevance_score'] = self.merged_data['avg_relevance_score'].fillna(0.5)
            self.merged_data['tag_list'] = self.merged_data['tag_list'].fillna('')
            
            logger.info(f"Merged dataset shape: {self.merged_data.shape}")
            logger.info(f"Columns: {self.merged_data.columns.tolist()}")
            
            return self.merged_data
        
        except Exception as e:
            logger.error(f"Error merging data: {str(e)}")
            return None
    
    def create_movie_features(self):
        """
        Create feature vectors for movies combining tags and genome scores.
        Uses multi-label binarization for tags.
        """
        try:
            logger.info("Creating movie feature matrix...")
            
            # Get unique movies
            unique_movies = self.merged_data[['movieId', 'title', 'genres', 'tag_list', 'avg_relevance_score']].drop_duplicates()
            
            # Extract all unique tags
            all_tags = set()
            for tags_list in unique_movies['tag_list']:
                if isinstance(tags_list, list):
                    all_tags.update(tags_list)
            
            logger.info(f"Total unique tags: {len(all_tags)}")
            
            # Create tag feature matrix
            movie_tag_features = []
            for idx, row in unique_movies.iterrows():
                tag_vector = np.zeros(len(all_tags))
                tags_list = row['tag_list']
                if isinstance(tags_list, list):
                    for tag in tags_list:
                        if tag in all_tags:
                            tag_vector[list(all_tags).index(tag)] = 1
                movie_tag_features.append(tag_vector)
            
            movie_tag_features = np.array(movie_tag_features)
            
            # Normalize relevance scores
            scaler = MinMaxScaler()
            relevance_scores = scaler.fit_transform(
                unique_movies['avg_relevance_score'].values.reshape(-1, 1)
            )
            
            # Combine features (tags + normalized relevance score)
            combined_features = np.hstack([
                movie_tag_features,
                relevance_scores * 2  # Weight relevance score
            ])
            
            logger.info(f"Movie feature matrix shape: {combined_features.shape}")
            
            return unique_movies, combined_features, list(all_tags)
        
        except Exception as e:
            logger.error(f"Error creating movie features: {str(e)}")
            return None, None, None


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
        logger.info(f"CollaborativeFilteringModel initialized with {n_factors} factors")
    
    def create_interaction_matrix(self, ratings_df):
        """Create user-item interaction matrix from ratings"""
        try:
            logger.info("Creating interaction matrix...")
            
            self.movie_ids = sorted(ratings_df['movieId'].unique())
            user_ids = sorted(ratings_df['userId'].unique())
            
            # Create pivot table
            interaction_matrix = ratings_df.pivot_table(
                index='userId',
                columns='movieId',
                values='rating',
                fill_value=0
            )
            
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
            
            # Find user factor (handle new users)
            if user_id <= len(self.user_factors):
                user_factor = self.user_factors[user_id - 1]
            else:
                user_factor = np.zeros(self.n_factors)
            
            # Compute dot product
            score = np.dot(user_factor, self.item_factors[movie_idx])
            return float(np.clip(score, 0, 5))
        
        except Exception as e:
            logger.error(f"Error in predict_score: {str(e)}")
            return 0


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
            logger.info(f"Similarity stats - Mean: {self.similarity_matrix.mean():.4f}, "
                       f"Max: {self.similarity_matrix.max():.4f}, Min: {self.similarity_matrix.min():.4f}")
            
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
            top_indices = np.argsort(similarities)[::-1][1:top_k+1]
            
            results = [
                (self.movie_ids[idx], float(similarities[idx]))
                for idx in top_indices
            ]
            
            return results
        
        except Exception as e:
            logger.error(f"Error in find_similar_movies: {str(e)}")
            return []


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
            logger.info("="*60)
            logger.info("Starting HybridRecommendationSystem Training")
            logger.info("="*60)
            
            # Step 1: Load and preprocess data
            logger.info("\n[Step 1] Loading and preprocessing data...")
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
            
            logger.info("\n" + "="*60)
            logger.info("Training Complete!")
            logger.info("="*60)
            
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
