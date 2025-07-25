from flask import Flask, render_template, request
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests

app = Flask(__name__)

# Load movie data
import os
data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'movies.csv'))

# Fill missing values and create combined features
features = ['genres', 'keywords', 'tagline', 'cast', 'overview']
for feature in features:
    data[feature] = data[feature].fillna('')
data['combine'] = (
    data['genres'] + ' ' +
    data['keywords'] + ' ' +
    data['tagline'] + ' ' +
    data['cast'] + ' ' +
    data['overview']
)

# Vectorize and calculate similarity
vector = TfidfVectorizer()
f_vector = vector.fit_transform(data['combine'])
similarity = cosine_similarity(f_vector)

def get_movie_details(title):
    api_key = '2d882462'
    url = f"http://www.omdbapi.com/?t={title}&apikey={api_key}"
    try:
        resp = requests.get(url)
        info = resp.json()
        if info.get('Response') == 'True':
            return {
                'poster': info.get('Poster', ''),
                'overview': info.get('Plot', 'Overview not available.'),
                'cast': info.get('Actors', 'Cast not available.'),
                'trailer': f"https://www.youtube.com/results?search_query={title.replace(' ', '+')}+official+trailer"
            }
    except Exception as e:
        print(f"Error fetching details for {title}: {e}")
    return None

@app.route('/')
def home():
    # initial empty state
    return render_template(
        'index.html',
        recommendations=[],
        movie_name="",
        message=""
    )

@app.route('/recommend', methods=['POST'])
def recommend():
    movie_name = request.form['movie'].strip()
    all_titles = data['title'].tolist()
    matches = difflib.get_close_matches(movie_name, all_titles)

    if not matches:
        return render_template(
            'index.html',
            recommendations=[],
            movie_name=movie_name,
            message="No close match found."
        )

    chosen = matches[0]
    idx = data[data.title == chosen].index[0]
    scores = list(enumerate(similarity[idx]))
    top = sorted(scores, key=lambda x: x[1], reverse=True)[1:11]

    recs = []
    for i, _ in top:
        title = data.at[i, 'title']
        details = get_movie_details(title)
        if details:
            recs.append({
                'title': title,
                'poster': details['poster'],
                'overview': details['overview'],
                'cast': details['cast'],
                'trailer': details['trailer']
            })

    return render_template(
        'index.html',
        recommendations=recs,
        movie_name=movie_name,
        message=""
    )

if __name__ == '__main__':
    app.run(debug=True)
