# 🎬 Movie Recommendation Web App

This is a simple and interactive movie recommendation web application built using **Flask** and **Python**, powered by **TF-IDF vectorization** and **cosine similarity** to recommend similar movies based on content features. It also integrates with the **OMDb API** to fetch posters, overviews, and cast details.

---

## 🚀 Features

- 🔍 **Search for any movie** by title  
- 🎯 **Get top 10 similar movie recommendations**  
- 🖼️ **Posters, cast info, and overviews**  
- ▶️ **Watch Trailer** (YouTube search)  
- 🌀 **Smooth UI with loading spinners**  
- 💡 **Autocomplete suggestions (optional enhancement)**

---

## 🛠️ Technologies Used

- **Frontend:** HTML, CSS, JavaScript (with spinner animation)  
- **Backend:** Python, Flask  
- **ML/NLP:** `TfidfVectorizer` from `scikit-learn`  
- **Data:** Movie metadata CSV  
- **API Integration:** OMDb API (https://www.omdbapi.com/)

---

## 📁 Project Structure
movie-recommender/
├── static/

├── templates/

│ └── index.html

├── app.py

├── movies.csv

└── README.md


---

## 🧠 How It Works

1. **Data Preprocessing**:
   - Combines genres, keywords, tagline, cast, and overview fields from `movies.csv`.
   - Uses **TF-IDF vectorization** to create numeric features.
   - Applies **cosine similarity** to find top 10 similar movies.

2. **Recommendation Logic**:
   - Takes a user input movie.
   - Finds the closest match from the dataset.
   - Returns the 10 most similar movies.

3. **OMDb API**:
   - Fetches poster, overview, and cast for each recommended movie.
   - Generates a YouTube search link for trailers.

---

## ✅ Setup Instructions

1. **Clone the repo**:
   ```bash
   git clone https://github.com/yourusername/movie-recommender.git
   cd movie-recommender
