# 🎬 Movie Recommendation System

A dual-engine movie recommender built on the **MovieLens dataset** that leverages both content-based and collaborative filtering to surface personalized film recommendations.

## Features

### Content-Based Filtering
- **Genre-based similarity**: Uses TF-IDF vectorization to measure genre similarity between movies
- **Fast lookups**: Cosine similarity matrix enables rapid recommendations
- **Fuzzy title matching**: Handles partial and misspelled movie titles with difflib

### Collaborative Filtering
- **User preference modeling**: SVD (Singular Value Decomposition) matrix factorization from scikit-surprise
- **Personalized recommendations**: Surfaces movies based on what similar users enjoy
- **Test-set evaluation**: Computes RMSE and MAE metrics on held-out data
- **Visualization**: Generates error distribution and prediction accuracy charts

### Interactive CLI
- Search for similar movies by title
- Get personalized picks for specific users
- Compare both recommendation methods side-by-side
- Run accuracy evaluations and export performance metrics

## Installation

### Prerequisites
- Python 3.7+

### Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scikit-surprise
```

Or install from `requirements.txt` (if available):
```bash
pip install -r requirements.txt
```

**Note**: `scikit-surprise` is required for collaborative filtering. Content-based filtering works without it.

## Usage

### Interactive Mode
```bash
python movie_recommendation_system.py
```

Choose from these options:
1. Find movies similar to a title (content-based)
2. Get personalized picks for a user (collaborative)
3. Compare both methods side-by-side
4. Run accuracy evaluation & export charts
5. Exit

### Command-Line Mode

#### Content-Based Recommendation
```bash
python movie_recommendation_system.py --mode content --title "Toy Story (1995)" --num 10
```

#### Collaborative Filtering Recommendation
```bash
python movie_recommendation_system.py --mode collaborative --user-id 1 --num 10
```

#### Compare Both Methods
```bash
python movie_recommendation_system.py --mode compare --user-id 1
```

#### Evaluate Model & Generate Charts
```bash
python movie_recommendation_system.py --mode evaluate
```

### Options
| Argument | Description |
|----------|-------------|
| `--interactive` | Start interactive CLI (default when no mode given) |
| `--mode {content,collaborative,compare,evaluate}` | Run a specific mode |
| `--title TEXT` | Movie title for content-based recommendations |
| `--user-id INT` | User ID for collaborative/comparison modes |
| `--num INT` | Number of recommendations to return (default: 10) |
| `--data-dir PATH` | Directory with MovieLens CSV files |
| `--verbose` | Enable debug logging |

## How It Works

### Content-Based Filtering
1. **TF-IDF Vectorization**: Converts pipe-separated genre strings into weighted word vectors
2. **Cosine Similarity**: Computes similarity scores between all movies
3. **Top-K Selection**: Uses efficient `argpartition` to avoid sorting the full matrix
4. **Returns**: Most similar movies ranked by genre overlap

### Collaborative Filtering
1. **SVD Training**: Decomposes the user-movie rating matrix into latent factors
2. **Train-Test Split**: Uses 75/25 split to validate model performance
3. **Prediction**: Estimates ratings for unseen movies per user
4. **Ranking**: Returns top-K predictions sorted by estimated rating
5. **Evaluation**: Computes RMSE and MAE on test set predictions

## Data

The system uses the **MovieLens Small Dataset** from GroupLens:
- **100,000 ratings** from 600+ users on 9,000+ movies
- **Automatically downloaded** on first run from: http://files.grouplens.org/datasets/movielens/ml-latest-small.zip
- **Format**: CSV files with movies, ratings, and metadata

## Output

### Visualizations (from `--mode evaluate`)
- **error_distribution.png**: Histogram of prediction errors with KDE
- **actual_vs_estimated.png**: Box plot comparing actual vs predicted ratings

### Metrics
- **RMSE**: Root Mean Squared Error (average magnitude of prediction errors)
- **MAE**: Mean Absolute Error (average absolute deviation)

## Example Session

```bash
$ python movie_recommendation_system.py

==================================================
   🎬  Movie Recommendation System
==================================================

  What would you like to do?
  1  Find movies similar to one I like  (by title)
  2  Get personalized picks for a user  (by user ID)
  3  Compare both methods side by side  (by user ID)
  4  Run accuracy evaluation & export charts
  5  Exit

  Your choice [1–5]: 1

  Movie title (e.g. 'Toy Story (1995)'): Toy Story (1995)

  Because you like 'Toy Story (1995)':
    1. Toy Story 2 (1999)
    2. Antz (1998)
    3. Bugs Life, A (1998)
    ...
```

## Architecture

### Core Class: `MovieRecommender`

**Data Members**:
- `movies_df`: Movie metadata (title, genres, movieId)
- `ratings_df`: User ratings history
- `tfidf_matrix`, `cosine_sim`: Content-based model
- `svd_model`, `predictions`: Collaborative filtering model

**Key Methods**:
- `load_data()`: Load MovieLens CSVs
- `train_content_based()`: Build TF-IDF similarity matrix
- `train_collaborative()`: Train SVD model
- `get_content_recommendations()`: Return genre-similar movies
- `get_collaborative_recommendations()`: Return personalized picks
- `evaluate_and_save_visualizations()`: Compute metrics & charts

## Performance Notes

- **Content-based**: O(N) per query using `argpartition`; no training time
- **Collaborative**: O(N×M×K) during SVD training where K = latent factors
- **Data loading**: Lazy evaluation—models train on first use

## Troubleshooting

### ImportError: No module named 'surprise'
Install scikit-surprise:
```bash
pip install scikit-surprise
```

### FileNotFoundError: CSV files not found
Ensure the MovieLens dataset is in the specified data directory, or delete the ZIP file to re-download:
```bash
rm ml-latest-small.zip
python movie_recommendation_system.py
```

### User ID not found
Verify the user ID exists in the dataset (typically 1–600 for MovieLens Small).

## Future Enhancements

- Hybrid recommendations combining content and collaborative signals
- Cold-start handling for new users/movies
- Real-time model updates
- Web API interface (Flask/FastAPI)
- Fine-tuned genre weights

## License

This project uses the MovieLens Small Dataset, which is provided by GroupLens research.

## References

- [MovieLens Dataset](https://grouplens.org/datasets/movielens/)
- [scikit-learn TF-IDF](https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting)
- [scikit-surprise SVD](http://surpriselib.com/)

---

Built with ❤️ for movie enthusiasts everywhere.
