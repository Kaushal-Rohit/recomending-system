"""
Movie Recommendation System using Hybrid Approach
This module implements a sophisticated recommendation engine that combines:
1. Collaborative Filtering (user-item interactions)
2. Content-Based Filtering (movie tags and features)
3. Relevance Scoring (genome-scores)
4. User Behavior Learning
"""

import logging
import pickle
import re
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer, StandardScaler, normalize

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

TAG_MIN_DF = 2
GENOME_SVD_COMPONENTS = 64
MIN_PROFILE_WEIGHT = 0.1
SIMILARITY_SCORE_SCALE = 5.0


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
        self.movie_metadata = None
        self.merged_data = None
        self.tag_vectorizer = None
        self.genre_binarizer = None
        self.genome_svd = None
        logger.info("DataPreprocessor initialized")

    @staticmethod
    def _extract_release_year(title: str) -> float:
        if not isinstance(title, str):
            return np.nan
        match = re.search(r'\((\d{4})\)\s*$', title)
        return float(match.group(1)) if match else np.nan
    
    def load_data(self):
        """Load all CSV files"""
        try:
            logger.info("Loading movies data...")
            self.movies = pd.read_csv(f'{self.data_dir}/movies.csv')
            self.movies['release_year'] = self.movies['title'].apply(self._extract_release_year)
            
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
            movie_tags = (
                self.tags.dropna(subset=['tag'])
                .groupby('movieId')['tag']
                .apply(lambda tags: sorted(set(tags.astype(str))))
                .reset_index()
            )
            movie_tags.columns = ['movieId', 'tag_list']
            movie_tags['tag_text'] = movie_tags['tag_list'].apply(lambda tags: ' '.join(tags))
            
            movies_with_tags = self.movies.merge(movie_tags, on='movieId', how='left')
            movies_with_tags['tag_list'] = movies_with_tags['tag_list'].apply(
                lambda tags: tags if isinstance(tags, list) else []
            )
            movies_with_tags['tag_text'] = movies_with_tags['tag_text'].fillna('')
            
            # Merge with genome scores
            genome_avg = self.genome_scores.groupby('movieId')['relevance'].mean().reset_index()
            genome_avg.columns = ['movieId', 'avg_relevance_score']
            
            self.movie_metadata = movies_with_tags.merge(
                genome_avg,
                on='movieId',
                how='left'
            )
            
            self.movie_metadata['avg_relevance_score'] = self.movie_metadata['avg_relevance_score'].fillna(0.5)
            self.movie_metadata['release_year'] = self.movie_metadata['release_year'].fillna(
                self.movie_metadata['release_year'].median()
            )
            
            # Merge with ratings
            self.merged_data = self.ratings.merge(
                self.movie_metadata,
                on='movieId',
                how='left'
            )
            
            # Fill missing values
            self.merged_data['avg_relevance_score'] = self.merged_data['avg_relevance_score'].fillna(0.5)
            self.merged_data['tag_list'] = self.merged_data['tag_list'].apply(
                lambda tags: tags if isinstance(tags, list) else []
            )
            self.merged_data['tag_text'] = self.merged_data['tag_text'].fillna('')
            
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
            if self.movie_metadata is None:
                logger.error("Movie metadata not available for feature creation")
                return None, None, None
            
            # Get unique movies with metadata
            unique_movies = self.movie_metadata[[
                'movieId', 'title', 'genres', 'tag_list', 'tag_text',
                'avg_relevance_score', 'release_year'
            ]].drop_duplicates()
            
            unique_movies['genre_list'] = unique_movies['genres'].fillna('').apply(
                lambda genres: genres.split('|') if genres else []
            )
            unique_movies['tag_text'] = unique_movies['tag_text'].fillna('')
            
            # Tag TF-IDF embeddings
            tag_corpus = unique_movies['tag_text']
            if tag_corpus.str.len().sum() == 0:
                tag_features = csr_matrix((len(unique_movies), 0))
                self.tag_vectorizer = None
                tags_list = []
            else:
                self.tag_vectorizer = TfidfVectorizer(min_df=TAG_MIN_DF, max_features=5000)
                tag_features = self.tag_vectorizer.fit_transform(tag_corpus)
                tags_list = self.tag_vectorizer.get_feature_names_out().tolist()
            
            # Genre multi-hot features
            self.genre_binarizer = MultiLabelBinarizer(sparse_output=True)
            genre_features = self.genre_binarizer.fit_transform(unique_movies['genre_list'])
            
            # Genome relevance embeddings (SVD on sparse relevance matrix)
            genome_embeddings = None
            if self.genome_scores is not None and not self.genome_scores.empty:
                movie_id_to_index = {
                    movie_id: idx for idx, movie_id in enumerate(unique_movies['movieId'].tolist())
                }
                movie_indices = self.genome_scores['movieId'].map(movie_id_to_index)
                valid_mask = movie_indices.notna()
                
                if valid_mask.any():
                    tag_id_to_index = {
                        tag_id: idx for idx, tag_id in enumerate(self.genome_scores['tagId'].unique())
                    }
                    rows = movie_indices[valid_mask].astype(int).to_numpy()
                    cols = self.genome_scores.loc[valid_mask, 'tagId'].map(tag_id_to_index).astype(int).to_numpy()
                    data = self.genome_scores.loc[valid_mask, 'relevance'].astype(float).to_numpy()
                    
                    genome_matrix = csr_matrix(
                        (data, (rows, cols)),
                        shape=(len(unique_movies), len(tag_id_to_index))
                    )
                    
                    if genome_matrix.shape[1] >= 2:
                        n_components = min(GENOME_SVD_COMPONENTS, genome_matrix.shape[1] - 1)
                        self.genome_svd = TruncatedSVD(
                            n_components=n_components,
                            random_state=42
                        )
                        genome_embeddings = self.genome_svd.fit_transform(genome_matrix)
            
            # Numeric features
            scaler = MinMaxScaler()
            relevance_scores = scaler.fit_transform(
                unique_movies['avg_relevance_score'].values.reshape(-1, 1)
            )
            release_year_scaled = scaler.fit_transform(
                unique_movies['release_year'].values.reshape(-1, 1)
            )
            
            feature_blocks = [
                tag_features,
                genre_features,
                csr_matrix(release_year_scaled),
                csr_matrix(relevance_scores)
            ]
            if genome_embeddings is not None:
                feature_blocks.append(csr_matrix(genome_embeddings))
            
            combined_features = hstack(feature_blocks, format='csr')
            combined_features = normalize(combined_features, norm='l2')
            
            logger.info(f"Movie feature matrix shape: {combined_features.shape}")
            
            return unique_movies, combined_features, tags_list
        
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
        self.user_ids = None
        self.user_id_to_index = {}
        self.movie_id_to_index = {}
        logger.info(f"CollaborativeFilteringModel initialized with {n_factors} factors")
    
    def create_interaction_matrix(self, ratings_df):
        """Create user-item interaction matrix from ratings"""
        try:
            logger.info("Creating interaction matrix...")
            self.movie_ids = sorted(ratings_df['movieId'].unique())
            self.user_ids = sorted(ratings_df['userId'].unique())
            self.user_id_to_index = {user_id: idx for idx, user_id in enumerate(self.user_ids)}
            self.movie_id_to_index = {movie_id: idx for idx, movie_id in enumerate(self.movie_ids)}
            
            rows = ratings_df['userId'].map(self.user_id_to_index).to_numpy()
            cols = ratings_df['movieId'].map(self.movie_id_to_index).to_numpy()
            data = ratings_df['rating'].astype(float).to_numpy()
            
            interaction_matrix = csr_matrix(
                (data, (rows, cols)),
                shape=(len(self.user_ids), len(self.movie_ids))
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
            if movie_id not in self.movie_id_to_index:
                return 0

            movie_idx = self.movie_id_to_index[movie_id]

            # Find user factor (handle new users)
            if user_id in self.user_id_to_index:
                user_factor = self.user_factors[self.user_id_to_index[user_id]]
            else:
                user_factor = np.zeros(self.n_factors)

            # Compute dot product
            score = np.dot(user_factor, self.item_factors[movie_idx])
            return float(np.clip(score, 0, 5))
        
        except Exception as e:
            logger.error(f"Error in predict_score: {str(e)}")
            return 0

    def predict_scores_for_user(self, user_id: int, movie_ids: List[int]) -> np.ndarray:
        """Predict ratings for a user across a list of movie IDs"""
        try:
            if user_id not in self.user_id_to_index:
                return np.zeros(len(movie_ids))

            user_idx = self.user_id_to_index[user_id]
            user_factor = self.user_factors[user_idx]
            all_scores = user_factor @ self.item_factors.T

            scores = np.zeros(len(movie_ids))
            for idx, movie_id in enumerate(movie_ids):
                movie_idx = self.movie_id_to_index.get(movie_id)
                if movie_idx is not None:
                    scores[idx] = all_scores[movie_idx]

            return np.clip(scores, 0, 5)

        except Exception as e:
            logger.error(f"Error predicting scores for user: {str(e)}")
            return np.zeros(len(movie_ids))


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
        self.movie_id_to_index = {}
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
            self.movie_id_to_index = {movie_id: idx for idx, movie_id in enumerate(self.movie_ids)}
            if hasattr(movie_features, "tocsr"):
                self.movie_features = movie_features.tocsr()
            else:
                self.movie_features = csr_matrix(movie_features)
            self.movie_features = normalize(self.movie_features, norm='l2')
            
            logger.info(f"Movie feature matrix shape: {self.movie_features.shape}")
            
            return True
        
        except Exception as e:
            logger.error(f"Error fitting model: {str(e)}")
            return False
    
    def find_similar_movies(self, movie_id: int, top_k: int = 10) -> List[Tuple[int, float]]:
        """Find top-k similar movies based on content features"""
        try:
            if movie_id not in self.movie_id_to_index:
                return []
            
            movie_idx = self.movie_id_to_index[movie_id]
            query_vector = self.movie_features[movie_idx]
            similarities = query_vector.dot(self.movie_features.T).toarray().ravel()
            similarities[movie_idx] = -1
            
            top_indices = np.argsort(similarities)[::-1][:top_k]
            results = [(self.movie_ids[idx], float(similarities[idx])) for idx in top_indices]
            
            return results
        
        except Exception as e:
            logger.error(f"Error in find_similar_movies: {str(e)}")
            return []

    def build_user_profile(self, movie_ids: List[int], ratings: List[float]):
        """Build a weighted user profile vector from watched movies"""
        try:
            indices = []
            weights = []
            for movie_id, rating in zip(movie_ids, ratings):
                movie_idx = self.movie_id_to_index.get(movie_id)
                if movie_idx is not None:
                    indices.append(movie_idx)
                    weights.append(max(rating, MIN_PROFILE_WEIGHT))
            
            if not indices:
                return None
            
            weights = np.array(weights, dtype=float)
            weights = weights / weights.sum()
            weights_matrix = csr_matrix(weights.reshape(1, -1))
            profile = weights_matrix.dot(self.movie_features[indices])
            return profile
        
        except Exception as e:
            logger.error(f"Error building user profile: {str(e)}")
            return None

    def score_candidates(self, profile_vector, candidate_movie_ids: List[int]) -> np.ndarray:
        """Score candidate movies by cosine similarity to a profile vector"""
        try:
            if profile_vector is None:
                return np.zeros(len(candidate_movie_ids))
            
            profile_vector = normalize(profile_vector, norm='l2')
            similarities = profile_vector.dot(self.movie_features.T).toarray().ravel()
            scores = np.zeros(len(candidate_movie_ids))
            for idx, movie_id in enumerate(candidate_movie_ids):
                movie_idx = self.movie_id_to_index.get(movie_id)
                if movie_idx is not None:
                    scores[idx] = similarities[movie_idx]
            return scores
        
        except Exception as e:
            logger.error(f"Error scoring candidates: {str(e)}")
            return np.zeros(len(candidate_movie_ids))


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
        self.user_stats = None
        self.movie_stats = None
        self.movie_metadata = None
        self.global_avg_rating = None
        self.global_release_year = None
        logger.info("UserBehaviorModel initialized")
    
    def engineer_features(self, merged_data, movie_metadata):
        """
        Engineer features for user behavior prediction.
        
        Features created:
        1. user_rating_mean: Average rating given by user
        2. user_rating_std: Standard deviation of user's ratings
        3. user_num_ratings: Count of ratings given by user
        4. movie_avg_rating: Average rating received by movie
        5. movie_num_ratings: Count of ratings for movie
        6. relevance_score: Genome relevance score
        7. release_year: Movie release year (normalized)
        8. genre_count: Number of genres
        """
        try:
            logger.info("Engineering features for user behavior...")
            
            # Create copy to avoid warnings
            data = merged_data.copy()
            metadata = movie_metadata.copy()
            metadata['genre_count'] = metadata['genres'].fillna('').apply(
                lambda genres: len(genres.split('|')) if genres else 0
            )
            self.global_release_year = metadata['release_year'].median()
            metadata['release_year'] = metadata['release_year'].fillna(self.global_release_year)
            
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
            
            self.user_stats = user_stats.set_index('userId')
            self.movie_stats = movie_stats.set_index('movieId')
            self.movie_metadata = metadata.set_index('movieId')
            self.global_avg_rating = data['rating'].mean()
            
            # Merge features
            features = data.merge(user_stats, on='userId', how='left')
            features = features.merge(movie_stats, on='movieId', how='left')
            features = features.merge(
                metadata[['movieId', 'avg_relevance_score', 'release_year', 'genre_count']],
                on='movieId',
                how='left'
            )
            
            # Handle missing values
            features['user_rating_std'].fillna(0, inplace=True)
            features['movie_avg_rating'].fillna(self.global_avg_rating, inplace=True)
            features['movie_num_ratings'].fillna(1, inplace=True)
            features['avg_relevance_score'].fillna(0.5, inplace=True)
            features['release_year'].fillna(self.global_release_year, inplace=True)
            features['genre_count'].fillna(0, inplace=True)
            
            # Additional features
            features['rating_diff'] = features['rating'] - features['movie_avg_rating']
            features['user_movie_interaction'] = features['user_rating_mean'] * features['movie_avg_rating']
            
            # Normalize user ratings (0-1 scale)
            features['user_rating_mean_norm'] = features['user_rating_mean'] / 5.0
            
            self.feature_names = [
                'user_rating_mean', 'user_rating_std', 'user_num_ratings',
                'movie_avg_rating', 'movie_num_ratings', 'avg_relevance_score',
                'release_year', 'genre_count', 'rating_diff',
                'user_movie_interaction', 'user_rating_mean_norm'
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

    def predict_for_user(self, user_id: int, candidate_movie_ids: List[int]) -> np.ndarray:
        """Predict preference scores for a user over candidate movies"""
        try:
            if self.user_stats is None or self.movie_stats is None or self.movie_metadata is None:
                return np.zeros(len(candidate_movie_ids))

            if user_id in self.user_stats.index:
                user_row = self.user_stats.loc[user_id]
            else:
                user_row = pd.Series({
                    'user_rating_mean': self.global_avg_rating,
                    'user_rating_std': 0,
                    'user_num_ratings': 0
                })

            movie_stats = self.movie_stats.reindex(candidate_movie_ids)
            metadata = self.movie_metadata.reindex(candidate_movie_ids)

            features = pd.DataFrame(index=candidate_movie_ids)
            features['user_rating_mean'] = user_row['user_rating_mean']
            features['user_rating_std'] = user_row['user_rating_std']
            features['user_num_ratings'] = user_row['user_num_ratings']
            features['movie_avg_rating'] = movie_stats['movie_avg_rating'].fillna(self.global_avg_rating)
            features['movie_num_ratings'] = movie_stats['movie_num_ratings'].fillna(1)
            features['avg_relevance_score'] = metadata['avg_relevance_score'].fillna(0.5)
            features['release_year'] = metadata['release_year'].fillna(self.global_release_year)
            features['genre_count'] = metadata['genre_count'].fillna(0)
            features['rating_diff'] = features['user_rating_mean'] - features['movie_avg_rating']
            features['user_movie_interaction'] = features['user_rating_mean'] * features['movie_avg_rating']
            features['user_rating_mean_norm'] = features['user_rating_mean'] / 5.0

            X_scaled = self.scaler.transform(features[self.feature_names].values)
            predictions = self.model.predict(X_scaled)
            return np.clip(predictions, 0, 5)

        except Exception as e:
            logger.error(f"Error in predict_for_user: {str(e)}")
            return np.zeros(len(candidate_movie_ids))


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
            features_data = self.ub_model.engineer_features(
                self.merged_data,
                self.preprocessor.movie_metadata
            )
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
            watched_movie_ids = user_ratings['movieId'].unique().tolist()
            watched_movies = set(watched_movie_ids)
            
            logger.info(f"User has watched {len(watched_movies)} movies")
            if len(user_ratings) > 0:
                logger.info(f"User's average rating: {user_ratings['rating'].mean():.2f}")
            
            # Get all unrated movies
            all_movie_ids = self.unique_movies['movieId'].unique().tolist()
            candidate_movie_ids = [movie_id for movie_id in all_movie_ids if movie_id not in watched_movies]

            logger.info(f"Generating scores for {len(candidate_movie_ids)} unrated movies...")
            if not candidate_movie_ids:
                return []

            # 1. Collaborative Filtering scores
            cf_scores = self.cf_model.predict_scores_for_user(user_id, candidate_movie_ids)

            # 2. Content-Based scores
            if current_movie_id and current_movie_id in self.cb_model.movie_id_to_index:
                profile_vector = self.cb_model.movie_features[self.cb_model.movie_id_to_index[current_movie_id]]
            else:
                profile_vector = self.cb_model.build_user_profile(
                    watched_movie_ids, user_ratings['rating'].tolist()
                )
            cb_scores = (
                self.cb_model.score_candidates(profile_vector, candidate_movie_ids)
                * SIMILARITY_SCORE_SCALE
            )

            # 3. User Behavior scores
            ub_scores = self.ub_model.predict_for_user(user_id, candidate_movie_ids)

            # Hybrid score (weighted combination)
            hybrid_scores = (
                self.cf_weight * cf_scores +
                self.cb_weight * cb_scores +
                self.ub_weight * ub_scores
            )

            candidate_info = self.unique_movies.set_index('movieId').reindex(candidate_movie_ids)
            top_indices = np.argsort(hybrid_scores)[::-1][:num_recommendations]

            recommendations = []
            for idx in top_indices:
                movie_id = candidate_movie_ids[idx]
                movie_row = candidate_info.loc[movie_id]
                if movie_row.isna().all():
                    continue
                release_year = movie_row['release_year']
                recommendations.append({
                    'movieId': int(movie_id),
                    'title': movie_row['title'],
                    'release_year': int(release_year) if not pd.isna(release_year) else None,
                    'genres': movie_row['genres'],
                    'tags': movie_row['tag_list'],
                    'hybrid_score': float(hybrid_scores[idx]),
                    'cf_score': float(cf_scores[idx]),
                    'cb_score': float(cb_scores[idx]),
                    'ub_score': float(ub_scores[idx]),
                    'avg_relevance_score': float(movie_row['avg_relevance_score'])
                })
            
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
