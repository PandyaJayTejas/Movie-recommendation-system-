# -*- coding: utf-8 -*-
"""Movie Recommendation System

A dual-engine movie recommender built on the MovieLens dataset.
Supports Content-Based Filtering (TF-IDF on genres) and Collaborative
Filtering (Matrix Factorization via SVD).
"""

import os
import zipfile
import urllib.request
import argparse
import logging
import difflib
import heapq
from typing import List, Dict, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

try:
    from surprise import SVD, Dataset, Reader, accuracy
    from surprise.model_selection import train_test_split
    SURPRISE_AVAILABLE = True
except ImportError:
    SURPRISE_AVAILABLE = False


class MovieRecommender:
    """
    A dual-engine movie recommender that can find similar films by genre
    or surface personalized picks based on what users like you have enjoyed.

    Content-Based:   Measures genre similarity via TF-IDF cosine scores.
    Collaborative:   Models user preferences through SVD matrix factorization.
    """

    DATA_URL = "http://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    ZIP_FILE = "ml-latest-small.zip"
    DATA_DIR = "ml-latest-small"

    def __init__(self, data_dir: str = DATA_DIR):
        self.data_dir = data_dir
        self.movies_df: Optional[pd.DataFrame] = None
        self.ratings_df: Optional[pd.DataFrame] = None

        # Content-based model
        self.tfidf = None
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.title_to_idx: Dict[str, int] = {}

        # Collaborative filtering model
        self.svd_model = None
        self.predictions = None
        self.prediction_results: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Internal guards — centralizes repeated checks so each method
    # can stay focused on what it actually does.
    # ------------------------------------------------------------------

    def _require_surprise(self) -> None:
        """Raises a clear error if scikit-surprise isn't installed."""
        if not SURPRISE_AVAILABLE:
            raise ImportError(
                "Collaborative filtering needs 'scikit-surprise'.\n"
                "Install it with:  pip install scikit-surprise"
            )

    def _ensure_data_loaded(self) -> None:
        """Loads data lazily — skips the trip to disk if already in memory."""
        if self.movies_df is None or self.ratings_df is None:
            self.load_data()

    def _ensure_content_trained(self) -> None:
        """Trains the content-based model on first use."""
        if self.cosine_sim is None:
            self.train_content_based()

    def _ensure_collab_trained(self) -> None:
        """Trains the collaborative model on first use."""
        self._require_surprise()
        if self.svd_model is None:
            self.train_collaborative()

    # ------------------------------------------------------------------
    # Data acquisition
    # ------------------------------------------------------------------

    def download_and_extract_data(self) -> None:
        """Fetches the MovieLens small dataset from GroupLens if not already present."""
        if os.path.exists(self.data_dir):
            logging.info(f"Dataset already found at '{self.data_dir}'. Skipping download.")
            return

        if not os.path.exists(self.ZIP_FILE):
            logging.info("Downloading MovieLens dataset from GroupLens...")
            try:
                urllib.request.urlretrieve(self.DATA_URL, self.ZIP_FILE)
                logging.info("Download complete.")
            except Exception as e:
                logging.error(f"Could not download the dataset: {e}")
                raise

        logging.info(f"Unpacking {self.ZIP_FILE}...")
        try:
            with zipfile.ZipFile(self.ZIP_FILE, 'r') as zf:
                zf.extractall('.')
            logging.info("Dataset ready.")
        except Exception as e:
            logging.error(f"Extraction failed: {e}")
            raise

    def load_data(self) -> None:
        """Reads movies and ratings CSVs into memory and does light cleanup."""
        movies_path = os.path.join(self.data_dir, 'movies.csv')
        ratings_path = os.path.join(self.data_dir, 'ratings.csv')

        if not os.path.exists(movies_path) or not os.path.exists(ratings_path):
            raise FileNotFoundError(
                f"CSV files not found in '{self.data_dir}'. "
                "Run download_and_extract_data() first."
            )

        logging.info("Loading movies and ratings...")
        self.movies_df = pd.read_csv(movies_path)
        self.ratings_df = pd.read_csv(ratings_path)

        # Fill missing values rather than silently dropping rows
        self.movies_df['title'] = self.movies_df['title'].fillna('Unknown Title')
        self.movies_df['genres'] = self.movies_df['genres'].fillna('(no genres listed)')
        self.ratings_df.dropna(subset=['userId', 'movieId', 'rating'], inplace=True)

        logging.info(
            f"Loaded {len(self.movies_df):,} movies and {len(self.ratings_df):,} ratings."
        )

    # ------------------------------------------------------------------
    # Content-Based Filtering
    # ------------------------------------------------------------------

    def train_content_based(self) -> None:
        """
        Builds a genre-based TF-IDF similarity matrix.

        Pipe-separated genres (e.g. 'Action|Comedy') are treated as word
        bags so TF-IDF can weight each genre token per-movie.
        """
        self._ensure_data_loaded()

        logging.info("Training content-based model (TF-IDF on genres)...")
        genres_text = self.movies_df['genres'].str.replace('|', ' ', regex=False)

        self.tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf.fit_transform(genres_text)
        self.cosine_sim = linear_kernel(self.tfidf_matrix, self.tfidf_matrix)

        # Keep the first occurrence when titles are duplicated in the dataset
        seen: set = set()
        self.title_to_idx = {}
        for idx, title in enumerate(self.movies_df['title']):
            if title not in seen:
                self.title_to_idx[title] = idx
                seen.add(title)

        logging.info("Content-based similarity matrix ready.")

    def suggest_titles(self, query: str) -> List[str]:
        """
        Returns up to 5 title suggestions for a partial or misspelled query.
        Tries exact substring matching first, then falls back to fuzzy difflib.
        """
        if self.movies_df is None:
            return []

        # Case-insensitive substring match catches most partial titles
        matches = self.movies_df.loc[
            self.movies_df['title'].str.contains(query, case=False, na=False), 'title'
        ].tolist()
        if matches:
            return matches[:5]

        # Fuzzy fallback for typos ("Toy Stoory" → "Toy Story")
        return difflib.get_close_matches(
            query, list(self.title_to_idx.keys()), n=5, cutoff=0.4
        )

    def get_content_recommendations(self, title: str, n_recommendations: int = 10) -> pd.Series:
        """
        Returns the top N most genre-similar movies to a given title.

        Uses argpartition (O(N)) to avoid sorting the full similarity matrix,
        then does a cheap sort only on the top candidates.
        """
        self._ensure_content_trained()

        if title not in self.title_to_idx:
            raise KeyError(f"'{title}' wasn't found in the dataset.")

        idx = self.title_to_idx[title]
        sim_scores = self.cosine_sim[idx]

        # Grab top N+1 candidates without sorting the whole array
        k = min(len(sim_scores), n_recommendations + 1)
        top_indices = np.argpartition(sim_scores, -k)[-k:]
        top_indices = top_indices[np.argsort(sim_scores[top_indices])[::-1]]

        # Exclude the input title itself, then take exactly N results
        recommended = [i for i in top_indices if i != idx][:n_recommendations]
        return self.movies_df['title'].iloc[recommended]

    # ------------------------------------------------------------------
    # Collaborative Filtering
    # ------------------------------------------------------------------

    def train_collaborative(self, test_size: float = 0.25, random_state: int = 42) -> None:
        """
        Trains an SVD model on user-movie ratings.

        A 75/25 train-test split is used, and test-set predictions are
        retained so evaluate_and_save_visualizations() can run without
        re-training.
        """
        self._require_surprise()
        self._ensure_data_loaded()

        logging.info("Training collaborative filtering model (SVD)...")
        reader = Reader(rating_scale=(0.5, 5.0))
        data = Dataset.load_from_df(
            self.ratings_df[['userId', 'movieId', 'rating']], reader
        )

        trainset, testset = train_test_split(
            data, test_size=test_size, random_state=random_state
        )

        self.svd_model = SVD(random_state=random_state)
        self.svd_model.fit(trainset)

        self.predictions = self.svd_model.test(testset)
        self.prediction_results = pd.DataFrame(
            [(p.uid, p.iid, p.r_ui, p.est) for p in self.predictions],
            columns=['userId', 'movieId', 'actual', 'estimated']
        )
        # Positive error = overestimate, negative = underestimate
        self.prediction_results['error'] = (
            self.prediction_results['estimated'] - self.prediction_results['actual']
        )

        logging.info("Collaborative model trained successfully.")

    def get_collaborative_recommendations(
        self, user_id: int, n_recommendations: int = 10
    ) -> pd.DataFrame:
        """
        Predicts ratings for every movie the user hasn't seen yet
        and returns their highest-ranked picks.

        Uses a set for O(1) 'already rated?' lookups and heapq.nlargest
        to pull top K without sorting the full prediction list.
        """
        self._ensure_collab_trained()

        all_movie_ids = self.movies_df['movieId'].unique()

        # O(1) membership check vs. the user's rated history
        already_rated = set(
            self.ratings_df.loc[self.ratings_df['userId'] == user_id, 'movieId']
        )
        unseen_movies = [m for m in all_movie_ids if m not in already_rated]

        if not unseen_movies:
            logging.warning(
                f"User {user_id} has rated every movie in the dataset — nothing new to recommend."
            )
            return pd.DataFrame(columns=['title', 'genres'])

        preds = [self.svd_model.predict(user_id, m) for m in unseen_movies]
        top_preds = heapq.nlargest(n_recommendations, preds, key=lambda p: p.est)
        top_movie_ids = [p.iid for p in top_preds]

        result = self.movies_df[self.movies_df['movieId'].isin(top_movie_ids)].copy()

        # Restore the prediction-score ranking after the DataFrame filter
        rank_map = {movie_id: rank for rank, movie_id in enumerate(top_movie_ids)}
        result['rank'] = result['movieId'].map(rank_map)

        return result.sort_values('rank').drop(columns='rank')[['title', 'genres']]

    # ------------------------------------------------------------------
    # Evaluation & Visualization
    # ------------------------------------------------------------------

    def evaluate_and_save_visualizations(self, save_dir: str = '.') -> Dict[str, float]:
        """
        Computes RMSE and MAE on the held-out test set and saves two charts:
          - error_distribution.png   : histogram of prediction errors
          - actual_vs_estimated.png  : box plot of estimated vs actual ratings
        """
        self._ensure_collab_trained()

        rmse = accuracy.rmse(self.predictions, verbose=False)
        mae = accuracy.mae(self.predictions, verbose=False)
        metrics = {'RMSE': rmse, 'MAE': mae}

        # --- Error distribution ---
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(
            self.prediction_results['error'], kde=True, bins=30, color='skyblue', ax=ax
        )
        ax.set_title('Prediction Error Distribution (Estimated − Actual)')
        ax.set_xlabel('Error')
        ax.set_ylabel('Count')
        ax.axvline(x=0, color='red', linestyle='--', label='Zero error')
        ax.legend()
        error_path = os.path.join(save_dir, 'error_distribution.png')
        fig.savefig(error_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        # --- Estimated vs actual ratings ---
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(
            x='actual', y='estimated',
            data=self.prediction_results,
            palette='Set3',
            ax=ax
        )
        ax.set_title('Estimated vs Actual Ratings')
        ax.set_xlabel('Actual Rating')
        ax.set_ylabel('Estimated Rating')
        ax.grid(axis='y', alpha=0.3)
        ratings_path = os.path.join(save_dir, 'actual_vs_estimated.png')
        fig.savefig(ratings_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        logging.info(f"Charts saved → {error_path}  |  {ratings_path}")
        return metrics

    def compare_recommendations(self, user_id: int) -> None:
        """Prints a side-by-side view of content-based vs collaborative picks for a user."""
        self._ensure_data_loaded()

        user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]
        if user_ratings.empty:
            print(f"No ratings found for User {user_id}.")
            return

        # Pick a highly rated film as the content-based seed; fall back to any rated film
        liked = user_ratings[user_ratings['rating'] >= 4.0].merge(self.movies_df, on='movieId')
        seed_df = liked if not liked.empty else user_ratings.merge(self.movies_df, on='movieId')
        seed_title = seed_df.iloc[0]['title']

        print(f"\n──── User {user_id}  •  Reference film: {seed_title} ────")

        print(f"\n[Content-Based]  Films similar in genre to '{seed_title}':")
        try:
            for i, title in enumerate(self.get_content_recommendations(seed_title, 5), 1):
                print(f"  {i}. {title}")
        except Exception as e:
            print(f"  Couldn't generate content-based picks: {e}")

        print("\n[Collaborative]  Personalized picks based on your rating history:")
        try:
            for i, row in enumerate(
                self.get_collaborative_recommendations(user_id, 5).itertuples(), 1
            ):
                print(f"  {i}. {row.title}  ({row.genres})")
        except Exception as e:
            print(f"  Couldn't generate collaborative picks: {e}")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def _print_collab_unavailable() -> None:
    """Single place to print the scikit-surprise install hint."""
    print("\n  Collaborative filtering requires scikit-surprise.")
    print("  Install it with:  pip install scikit-surprise\n")


def interactive_loop(recommender: MovieRecommender) -> None:
    """Interactive command-line interface for the recommendation engine."""
    print("\n" + "=" * 55)
    print("   🎬  Movie Recommendation System")
    print("=" * 55)

    while True:
        print("\n  What would you like to do?")
        print("  1  Find movies similar to one I like  (by title)")
        print("  2  Get personalized picks for a user  (by user ID)")
        print("  3  Compare both methods side by side  (by user ID)")
        print("  4  Run accuracy evaluation & export charts")
        print("  5  Exit")

        choice = input("\n  Your choice [1–5]: ").strip()

        if choice == '1':
            title = input("\n  Movie title (e.g. 'Toy Story (1995)'): ").strip()
            if not title:
                continue
            try:
                recs = recommender.get_content_recommendations(title, 10)
                print(f"\n  Because you like '{title}':")
                for i, t in enumerate(recs, 1):
                    print(f"    {i}. {t}")
            except KeyError:
                print(f"\n  Couldn't find '{title}' in the dataset.")
                suggestions = recommender.suggest_titles(title)
                if suggestions:
                    print("  Did you mean one of these?")
                    for i, s in enumerate(suggestions, 1):
                        print(f"    {i}. {s}")
                else:
                    print("  No close matches found. Try a different search.")
            except Exception as e:
                print(f"  Error: {e}")

        elif choice in ('2', '3', '4'):
            # All three modes require scikit-surprise — check once up front
            if not SURPRISE_AVAILABLE:
                _print_collab_unavailable()
                continue

            if choice in ('2', '3'):
                raw = input("\n  User ID (number): ").strip()
                if not raw.isdigit():
                    print("  Please enter a valid integer User ID.")
                    continue
                u_id = int(raw)

                if choice == '2':
                    try:
                        recs = recommender.get_collaborative_recommendations(u_id, 10)
                        print(f"\n  Personalized picks for User {u_id}:")
                        if recs.empty:
                            print("  No new recommendations available.")
                        else:
                            for i, row in enumerate(recs.itertuples(), 1):
                                print(f"    {i}. {row.title}  |  {row.genres}")
                    except Exception as e:
                        print(f"  Error: {e}")

                else:  # choice == '3'
                    try:
                        recommender.compare_recommendations(u_id)
                    except Exception as e:
                        print(f"  Error: {e}")

            else:  # choice == '4'
                try:
                    print("\n  Training model and evaluating — this may take a moment...")
                    metrics = recommender.evaluate_and_save_visualizations()
                    print(f"\n  Results:")
                    print(f"    RMSE : {metrics['RMSE']:.4f}")
                    print(f"    MAE  : {metrics['MAE']:.4f}")
                    print("\n  Charts saved to the current directory:")
                    print("    • error_distribution.png")
                    print("    • actual_vs_estimated.png")
                except Exception as e:
                    print(f"  Evaluation failed: {e}")

        elif choice == '5':
            print("\n  See you next time! 🎬\n")
            break

        else:
            print("  Invalid choice. Please pick a number between 1 and 5.")


def main():
    parser = argparse.ArgumentParser(
        description="Movie Recommendation System — content-based and collaborative"
    )
    parser.add_argument('--interactive', action='store_true',
                        help="Start the interactive CLI (default when no mode is given)")
    parser.add_argument('--mode', choices=['content', 'collaborative', 'compare', 'evaluate'],
                        help="Run a specific mode non-interactively")
    parser.add_argument('--title', type=str,
                        help="Movie title for content-based recommendations")
    parser.add_argument('--user-id', type=int,
                        help="User ID for collaborative or comparison recommendations")
    parser.add_argument('--num', type=int, default=10,
                        help="Number of recommendations to return (default: 10)")
    parser.add_argument('--data-dir', type=str, default=MovieRecommender.DATA_DIR,
                        help="Directory containing the MovieLens CSV files")
    parser.add_argument('--verbose', action='store_true',
                        help="Enable detailed debug logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s  %(levelname)-8s  %(message)s'
    )

    recommender = MovieRecommender(data_dir=args.data_dir)

    try:
        recommender.download_and_extract_data()
        recommender.load_data()
    except Exception as e:
        logging.critical(f"Startup failed: {e}")
        return

    if args.mode == 'content':
        if not args.title:
            parser.error("--title is required with --mode content")
        try:
            recs = recommender.get_content_recommendations(args.title, args.num)
            print(f"\nTop {args.num} similar to '{args.title}':")
            for i, t in enumerate(recs, 1):
                print(f"  {i}. {t}")
        except KeyError:
            print(f"'{args.title}' not found.")
            suggestions = recommender.suggest_titles(args.title)
            if suggestions:
                print("Did you mean:")
                for i, s in enumerate(suggestions, 1):
                    print(f"  {i}. {s}")

    elif args.mode == 'collaborative':
        if args.user_id is None:
            parser.error("--user-id is required with --mode collaborative")
        if not SURPRISE_AVAILABLE:
            print("Error: scikit-surprise is not installed.")
            return
        try:
            recs = recommender.get_collaborative_recommendations(args.user_id, args.num)
            print(f"\nTop {args.num} picks for User {args.user_id}:")
            for i, row in enumerate(recs.itertuples(), 1):
                print(f"  {i}. {row.title}  |  {row.genres}")
        except Exception as e:
            print(f"Error: {e}")

    elif args.mode == 'compare':
        if args.user_id is None:
            parser.error("--user-id is required with --mode compare")
        if not SURPRISE_AVAILABLE:
            print("Error: scikit-surprise is not installed.")
            return
        recommender.compare_recommendations(args.user_id)

    elif args.mode == 'evaluate':
        if not SURPRISE_AVAILABLE:
            print("Error: scikit-surprise is not installed.")
            return
        try:
            metrics = recommender.evaluate_and_save_visualizations()
            print(f"\nCollaborative Filtering Metrics:")
            print(f"  RMSE : {metrics['RMSE']:.4f}")
            print(f"  MAE  : {metrics['MAE']:.4f}")
        except Exception as e:
            print(f"Evaluation failed: {e}")

    else:
        interactive_loop(recommender)


if __name__ == '__main__':
    main()
