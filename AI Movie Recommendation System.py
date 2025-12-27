import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from colorama import init, Fore
import time
import sys

# =========================
# Initialize colorama
# =========================
init(autoreset=True)

# =========================
# Load and preprocess data
# =========================
def load_data(file_path='imdb_top_1000.csv'):
    try:
        df = pd.read_csv(file_path)
        df['combined_features'] = (
            df['Genre'].fillna('') + ' ' + df['Overview'].fillna('')
        )
        return df
    except FileNotFoundError:
        print(Fore.RED + f"Error: The file '{file_path}' was not found.")
        sys.exit()

movies_df = load_data()

# =========================
# TF-IDF Vectorization
# =========================
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_df['combined_features'])

# =========================
# Cosine Similarity
# =========================
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# =========================
# List Unique Genres
# =========================
def list_genres(df):
    return sorted(
        set(
            genre.strip()
            for sublist in df['Genre'].dropna().str.split(', ')
            for genre in sublist
        )
    )

genres = list_genres(movies_df)

# =========================
# Processing Animation
# =========================
def processing_animation(message="Processing"):
    for _ in range(3):
        for dot in ". . .".split():
            sys.stdout.write(Fore.YELLOW + message + dot + "\r")
            sys.stdout.flush()
            time.sleep(0.4)
    print(" " * 30, end="\r")

# =========================
# Movie Recommendation Logic
# =========================
def recommend_movies(genre=None, mood=None, rating=None, top_n=5):
    filtered_df = movies_df

    if genre:
        filtered_df = filtered_df[
            filtered_df['Genre'].str.contains(genre, case=False, na=False)
        ]

    if rating:
        filtered_df = filtered_df[
            filtered_df['IMDB_Rating'] >= rating
        ]

    filtered_df = filtered_df.sample(frac=1).reset_index(drop=True)

    recommendations = []

    for _, row in filtered_df.iterrows():
        overview = row['Overview']
        if pd.isna(overview):
            continue

        movie_polarity = TextBlob(overview).sentiment.polarity

        if mood:
            mood_polarity = TextBlob(mood).sentiment.polarity
            if (mood_polarity < 0 and movie_polarity > 0) or movie_polarity >= 0:
                recommendations.append((row['Series_Title'], movie_polarity))
        else:
            recommendations.append((row['Series_Title'], movie_polarity))

        if len(recommendations) == top_n:
            break

    return recommendations

# =========================
# Display Recommendations
# =========================
def display_recommendations(recs, name):
    print(Fore.YELLOW + f"\n🍿 AI Movie Recommendations for {name}:\n")

    for idx, (title, polarity) in enumerate(recs, 1):
        if polarity > 0:
            sentiment = "Positive 😊"
        elif polarity < 0:
            sentiment = "Negative 😞"
        else:
            sentiment = "Neutral 😐"

        print(
            f"{Fore.CYAN}{idx}. 🎥 {title} "
            f"(Polarity: {polarity:.2f}, {sentiment})"
        )

# =========================
# Handle AI Recommendation
# =========================
def handle_ai():
    name = input(Fore.GREEN + "Enter your name: ").strip() or "User"

    print(Fore.CYAN + "\nAvailable Genres:")
    print(", ".join(genres))

    genre = input(Fore.GREEN + "\nPreferred genre (optional): ").strip()
    mood = input(Fore.GREEN + "How are you feeling today? (optional): ").strip()

    try:
        rating = float(input(Fore.GREEN + "Minimum IMDb rating (optional): ").strip())
    except ValueError:
        rating = None

    try:
        top_n = int(input(Fore.GREEN + "Number of recommendations: ").strip())
    except ValueError:
        top_n = 5

    processing_animation("Analyzing mood")
    processing_animation("Finding movies")

    recs = recommend_movies(genre, mood, rating, top_n)

    if recs:
        display_recommendations(recs, name)
    else:
        print(Fore.RED + "No suitable recommendations found.")


def main():
    print(Fore.MAGENTA + "🎬 Welcome to AI Movie Recommendation System 🎬")
    handle_ai()

if __name__ == "__main__":
    main()
