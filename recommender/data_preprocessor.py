import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer

from .logging_config import get_logger


logger = get_logger(__name__)


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

            logger.info(
                f"Movies: {self.movies.shape}, Tags: {self.tags.shape}, "
                f"Ratings: {self.ratings.shape}, Genome: {self.genome_scores.shape}"
            )

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
            unique_movies = self.merged_data[
                ['movieId', 'title', 'genres', 'tag_list', 'avg_relevance_score']
            ].drop_duplicates()

            unique_movies['tag_list'] = unique_movies['tag_list'].apply(
                lambda tags: tags if isinstance(tags, list) else []
            )
            tag_lists = unique_movies['tag_list']

            # Create tag feature matrix
            tag_binarizer = MultiLabelBinarizer()
            movie_tag_features = tag_binarizer.fit_transform(tag_lists)
            all_tags = tag_binarizer.classes_.tolist()

            logger.info(f"Total unique tags: {len(all_tags)}")

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
