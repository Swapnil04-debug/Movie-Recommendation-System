import os
import re
import ast
import json
import time
import logging
import threading
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher

import requests
import pandas as pd

from requests.adapters import HTTPAdapter
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/"

TMDB_READ_ACCESS_TOKEN = os.getenv("TMDB_READ_ACCESS_TOKEN", "").strip()
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "").strip()
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "").strip()

REQUEST_TIMEOUT = (5, 10)
MAX_ATTEMPTS = 3
RETRY_STATUS = (429, 500, 502, 503, 504)

CACHE_DURATION = 1800            # normal responses: 30 minutes
GENRE_CACHE_DURATION = 86400     # genre list rarely changes: 24 hours
FAILURE_CACHE_DURATION = 5       # CHANGED: failures remembered only briefly
MAX_CACHE_ENTRIES = 1000         # keeps memory bounded
REFRESH_COOLDOWN = 60            # seconds between forced home refreshes

FUZZY_THRESHOLD = 0.85           # typo tolerance for title matching
MAX_FRANCHISE = 6                # max same-franchise movies shown first
MIN_LOCAL_VOTES = 50             # ignore obscure CSV movies as recommendations
MIN_LOCAL_SIMILARITY = 0.05
CAST_LIMIT = 10
IMPORTANT_JOBS = {"Director", "Screenplay", "Writer", "Story"}

# Local CSV rows without a poster get one from TMDB (cached, parallel).
HYDRATE_LOCAL_POSTERS = os.getenv("HYDRATE_LOCAL_POSTERS", "1") == "1"

CSV_PATH = os.path.join(BASE_DIR, "movies.csv")

session = requests.Session()
_adapter = HTTPAdapter(pool_connections=10, pool_maxsize=20)
session.mount("https://", _adapter)
session.mount("http://", _adapter)


# ============================================================
# TMDB AUTHENTICATION
# ============================================================

def tmdb_enabled():
    return bool(TMDB_READ_ACCESS_TOKEN or TMDB_API_KEY)


def tmdb_headers():
    headers = {
        "accept": "application/json",
        "User-Agent": "CinemaScope/1.0"
    }

    if TMDB_READ_ACCESS_TOKEN:
        headers["Authorization"] = f"Bearer {TMDB_READ_ACCESS_TOKEN}"

    return headers


# ============================================================
# THREAD-SAFE CACHE
# ============================================================

_MISS = object()
_api_cache = {}
_cache_lock = threading.Lock()
_inflight = {}


def get_cache(key):
    """Return the cached value, or _MISS. None is a valid cached value."""
    with _cache_lock:
        entry = _api_cache.get(key)

        if entry is None:
            return _MISS

        value, expires = entry

        if time.time() > expires:
            _api_cache.pop(key, None)
            return _MISS

        return value


def set_cache(key, value, ttl=CACHE_DURATION):
    with _cache_lock:
        if len(_api_cache) >= MAX_CACHE_ENTRIES:
            now = time.time()

            for old_key in [
                k for k, (_, exp) in _api_cache.items() if exp < now
            ]:
                _api_cache.pop(old_key, None)

            while len(_api_cache) >= MAX_CACHE_ENTRIES:
                _api_cache.pop(next(iter(_api_cache)))

        _api_cache[key] = (value, time.time() + ttl)


def _key_lock(key):
    with _cache_lock:
        return _inflight.setdefault(key, threading.Lock())


# ============================================================
# GENERAL HELPERS
# ============================================================

def normalize_title(title):
    """
    Lowercase, strip accents and keep only letters/digits
    (non-Latin scripts are preserved).
    """
    if not title:
        return ""

    text = unicodedata.normalize("NFKD", str(title).lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))

    return re.sub(r"[\W_]+", "", text)


def canonical_title(title):
    """Normalize a title while ignoring a leading article."""
    raw_title = str(title or "").strip().lower()
    raw_title = re.sub(r"^(?:the|a|an)\s+", "", raw_title)

    return normalize_title(raw_title)


def title_keys(movie):
    """Canonical keys for a TMDB movie's title and original title."""
    return {
        key for key in (
            canonical_title(movie.get("title")),
            canonical_title(movie.get("original_title"))
        ) if key
    }


def clean_str(value):
    """Convert NaN/None to an empty string."""
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value).strip()


def tmdb_image(path, size="w500"):
    path = clean_str(path)

    if not path:
        return None

    if path.startswith("http"):
        return path

    return f"{TMDB_IMAGE_BASE_URL}{size}{path}"


def parse_list(value):
    """
    Parse CSV cells holding JSON or Python-literal lists.
    Returns [] when the value is not a list.
    """
    if isinstance(value, list):
        return value

    if not isinstance(value, str) or not value.strip():
        return []

    text = value.strip()

    for loader in (json.loads, ast.literal_eval):
        try:
            parsed = loader(text)
            return parsed if isinstance(parsed, list) else []
        except (ValueError, SyntaxError, TypeError, MemoryError,
                RecursionError):
            continue

    return []


def extract_names(value, limit=None, jobs=None):
    """
    Return a list of names from a CSV cell.

    Supports JSON/literal lists of dicts ({"name": ...}), lists of
    strings, and plain "A, B | C" strings. `jobs` filters crew dicts.
    """
    parsed = parse_list(value)
    names = []

    if parsed:
        for item in parsed:
            if isinstance(item, dict):
                if jobs and item.get("job") not in jobs:
                    continue
                name = item.get("name")
            elif isinstance(item, str):
                name = item
            else:
                name = None

            if name and str(name).strip():
                names.append(str(name).strip())

    elif isinstance(value, str) and value.strip():
        names = [
            part.strip()
            for part in re.split(r"[|,;]", value)
            if part.strip()
        ]

    return names[:limit] if limit else names


def extract_genres(value):
    names = extract_names(value)
    return ", ".join(names) if names else "Movie"


def safe_int(value):
    try:
        if pd.isna(value):
            return None

        return int(float(value))

    except (ValueError, TypeError):
        return None


def safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return default

        return float(value)

    except (ValueError, TypeError):
        return default


def movie_key(movie):
    """Identity for de-duplication: title AND year (remakes differ)."""
    return (
        canonical_title(movie.get("title", "")),
        str(movie.get("year", "") or "")
    )


# ============================================================
# TMDB REQUEST WITH CACHE, RETRIES AND STAMPEDE PROTECTION
# ============================================================

def _retry_delay(response, attempt):
    try:
        return min(float(response.headers.get("Retry-After")), 5.0)
    except (TypeError, ValueError):
        return attempt * 1.5


def _fetch_tmdb(url, endpoint, params):
    params = dict(params)

    if not TMDB_READ_ACCESS_TOKEN and TMDB_API_KEY:
        params["api_key"] = TMDB_API_KEY

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = session.get(
                url,
                headers=tmdb_headers(),
                params=params,
                timeout=REQUEST_TIMEOUT
            )

            if (
                response.status_code in RETRY_STATUS
                and attempt < MAX_ATTEMPTS
            ):
                logger.warning(
                    "TMDB returned HTTP %s for %s (attempt %s/%s).",
                    response.status_code, endpoint, attempt, MAX_ATTEMPTS
                )
                time.sleep(_retry_delay(response, attempt))
                continue

            response.raise_for_status()
            return response.json()

        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout
        ) as exc:
            logger.warning(
                "TMDB connection failed for %s (attempt %s/%s): %s",
                endpoint, attempt, MAX_ATTEMPTS, exc
            )

            if attempt < MAX_ATTEMPTS:
                time.sleep(attempt * 1.5)

        except requests.exceptions.HTTPError as exc:
            logger.error("TMDB HTTP error for %s: %s", endpoint, exc)
            return None

        except (requests.exceptions.RequestException, ValueError) as exc:
            logger.error("TMDB request failed for %s: %s", endpoint, exc)
            return None

    logger.error(
        "TMDB request failed after %s attempts: %s",
        MAX_ATTEMPTS, endpoint
    )
    return None


def tmdb_request(endpoint, params=None, refresh=False, ttl=None):
    """
    Cached TMDB request.

    - Identical concurrent requests share one network call.
    - Failures are cached very briefly so an outage is not hammered.
    - refresh=True skips reading the cache but still stores the result.
    """
    if not tmdb_enabled():
        logger.warning("TMDB credentials are not configured.")
        return None

    params = dict(params or {})
    endpoint = endpoint.lstrip("/")
    url = f"{TMDB_BASE_URL}/{endpoint}"
    cache_key = f"{endpoint}?{sorted(params.items())}"

    if not refresh:
        cached = get_cache(cache_key)

        if cached is not _MISS:
            return cached

    lock = _key_lock(cache_key)

    try:
        with lock:
            if not refresh:
                cached = get_cache(cache_key)

                if cached is not _MISS:
                    return cached

            result = _fetch_tmdb(url, endpoint, params)

            set_cache(
                cache_key,
                result,
                (ttl or CACHE_DURATION)
                if result is not None
                else FAILURE_CACHE_DURATION
            )

            return result

    finally:
        with _cache_lock:
            _inflight.pop(cache_key, None)


# ============================================================
# GENRE MAPPING
# ============================================================

def get_genre_map():
    response = tmdb_request(
        "genre/movie/list",
        params={"language": "en"},
        ttl=GENRE_CACHE_DURATION
    )

    if not response:
        return {}

    return {
        genre["id"]: genre["name"]
        for genre in response.get("genres", [])
    }


def genre_names_from_ids(genre_ids):
    if not isinstance(genre_ids, list) or not genre_ids:
        return "Movie"

    genre_map = get_genre_map()

    names = [
        genre_map[genre_id]
        for genre_id in genre_ids
        if genre_id in genre_map
    ]

    return ", ".join(names) if names else "Movie"


# ============================================================
# NORMALIZE TMDB MOVIE
# ============================================================

def format_movie(movie):
    if not movie:
        return None

    release_date = movie.get("release_date") or ""
    year = release_date[:4] if len(release_date) >= 4 else "N/A"

    genre_ids = movie.get("genre_ids", [])

    if not genre_ids and movie.get("genres"):
        genres = [
            genre.get("name")
            for genre in movie.get("genres", [])
            if genre.get("name")
        ]
        genres_text = ", ".join(genres) if genres else "Movie"
    else:
        genres_text = genre_names_from_ids(genre_ids)

    rating = safe_float(movie.get("vote_average"))

    return {
        "id": movie.get("id"),
        "source": "tmdb",
        "title": (
            movie.get("title")
            or movie.get("original_title")
            or "Unknown title"
        ),
        "year": year,
        "release_date": release_date,
        "rating": round(rating, 1),
        "vote_average": round(rating, 1),
        "vote_count": movie.get("vote_count", 0),
        "popularity": movie.get("popularity", 0),
        "genres": genres_text,
        "genre_ids": genre_ids,
        "overview": movie.get("overview") or "",
        "poster": tmdb_image(movie.get("poster_path")),
        "backdrop": tmdb_image(movie.get("backdrop_path"), "w1280"),
        "poster_path": movie.get("poster_path"),
        "backdrop_path": movie.get("backdrop_path"),
        "similarity": None
    }


# ============================================================
# HOME PAGE MOVIE COLLECTIONS
# ============================================================

HOME_COLLECTIONS = {
    "trending_today": (
        "trending/movie/day",
        {"language": "en-US", "page": 1}
    ),
    "trending_week": (
        "trending/movie/week",
        {"language": "en-US", "page": 1}
    ),
    "popular": (
        "movie/popular",
        {"language": "en-US", "region": "IN", "page": 1}
    ),
    "now_playing": (
        "movie/now_playing",
        {"language": "en-US", "region": "IN", "page": 1}
    ),
    "upcoming": (
        "movie/upcoming",
        {"language": "en-US", "region": "IN", "page": 1}
    ),
    "top_rated": (
        "movie/top_rated",
        {"language": "en-US", "page": 1}
    )
}

_last_home_refresh = 0.0


def get_tmdb_list(endpoint, params=None, limit=20, refresh=False):
    result = tmdb_request(
        endpoint,
        params=params or {"language": "en-US", "page": 1},
        refresh=refresh
    )

    if not result:
        return []

    movies = []

    for movie in result.get("results", [])[:limit]:
        formatted = format_movie(movie)

        if formatted:
            movies.append(formatted)

    return movies


def get_home_data(force_refresh=False):
    """
    Fetch all live movie sections concurrently.

    Forced refreshes are rate-limited, and an all-empty result
    (TMDB outage) is only cached very briefly.
    """
    global _last_home_refresh

    cache_key = "cinemascope_home"

    if force_refresh:
        if time.time() - _last_home_refresh < REFRESH_COOLDOWN:
            force_refresh = False
        else:
            _last_home_refresh = time.time()

    if not force_refresh:
        cached = get_cache(cache_key)

        if cached is not _MISS and cached:
            return cached

    empty_data = {
        "featured": None,
        "trending_today": [],
        "trending_week": [],
        "popular": [],
        "now_playing": [],
        "upcoming": [],
        "top_rated": []
    }

    if not tmdb_enabled():
        return empty_data

    def fetch_collection(item):
        key, (endpoint, params) = item

        return key, get_tmdb_list(
            endpoint,
            params=params,
            limit=20,
            refresh=force_refresh
        )

    home_data = dict(empty_data)

    with ThreadPoolExecutor(max_workers=6) as executor:
        for key, movies in executor.map(
            fetch_collection,
            HOME_COLLECTIONS.items()
        ):
            home_data[key] = movies

    if not home_data["trending_today"]:
        home_data["trending_today"] = home_data["popular"]

    if not home_data["trending_week"]:
        home_data["trending_week"] = home_data["popular"]

    if not home_data["popular"]:
        home_data["popular"] = home_data["trending_week"]

    featured_candidates = (
        home_data["trending_today"]
        or home_data["popular"]
        or home_data["now_playing"]
    )

    home_data["featured"] = (
        featured_candidates[0] if featured_candidates else None
    )

    has_data = any(
        home_data[key] for key in HOME_COLLECTIONS
    )

    set_cache(
        cache_key,
        home_data,
        CACHE_DURATION if has_data else FAILURE_CACHE_DURATION
    )

    return home_data


# ============================================================
# LOCAL CSV DATASET AND TF-IDF
# ============================================================

data = pd.DataFrame()
tfidf_matrix = None
tfidf_vectorizer = None

FEATURE_COLUMNS = [
    "genres",
    "keywords",
    "tagline",
    "cast",
    "crew",
    "overview"
]


def build_feature_text(
    genres=(),
    keywords=(),
    tagline="",
    overview="",
    cast=(),
    directors=()
):
    """
    Build a clean text profile used by BOTH the CSV rows and the
    TMDB query movie, so the two are directly comparable.

    Names and multi-word genres/keywords are collapsed into single
    tokens ("Robert Downey Jr." -> "robertdowneyjr") so that actors
    match as a whole instead of matching every "Robert".
    Genres, keywords and directors are repeated to weight them higher.
    """

    def token(value):
        return normalize_title(value)

    parts = []

    parts += [token(g) for g in genres] * 2
    parts += [token(k) for k in keywords] * 2
    parts += [token(d) for d in directors] * 2
    parts += [token(c) for c in cast]

    parts.append(clean_str(tagline))
    parts.append(clean_str(overview))

    return " ".join(part for part in parts if part)


def csv_row_features(row):
    return build_feature_text(
        genres=extract_names(row["genres"]),
        keywords=extract_names(row["keywords"]),
        tagline=row["tagline"],
        overview=row["overview"],
        cast=extract_names(row["cast"], limit=CAST_LIMIT),
        directors=extract_names(row["crew"], jobs=IMPORTANT_JOBS)
    )


def load_movie_dataset():
    global data
    global tfidf_matrix
    global tfidf_vectorizer

    if not os.path.exists(CSV_PATH):
        logger.warning("movies.csv was not found at %s", CSV_PATH)
        return

    try:
        frame = pd.read_csv(CSV_PATH)
        frame.columns = frame.columns.str.strip()

        if "title" not in frame.columns:
            logger.error("movies.csv must contain a title column.")
            return

        frame = frame.dropna(subset=["title"]).reset_index(drop=True)
        frame["title"] = frame["title"].astype(str).str.strip()

        for column in FEATURE_COLUMNS:
            if column not in frame.columns:
                frame[column] = ""

            frame[column] = frame[column].fillna("")

        # Precomputed once instead of on every request.
        frame["title_key"] = frame["title"].map(canonical_title)

        id_column = next(
            (c for c in ("movie_id", "id") if c in frame.columns),
            None
        )

        frame["_id"] = (
            pd.to_numeric(frame[id_column], errors="coerce")
            if id_column else float("nan")
        )

        frame["_votes"] = (
            pd.to_numeric(frame["vote_count"], errors="coerce")
            if "vote_count" in frame.columns else float("nan")
        )

        frame["combined_features"] = frame.apply(
            csv_row_features,
            axis=1
        )

        # Genre sets used by the genre-based fallback recommender.
        frame["_genre_set"] = frame["genres"].map(
            lambda value: {
                normalize_title(name) for name in extract_names(value)
            }
        )

        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=20000,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True
        )

        matrix = vectorizer.fit_transform(frame["combined_features"])

        data = frame
        tfidf_vectorizer = vectorizer
        tfidf_matrix = matrix

        logger.info("Loaded %s movies from movies.csv.", len(data))
        logger.info("TF-IDF matrix shape: %s", tfidf_matrix.shape)

    except Exception as exc:
        logger.exception("Could not load movie dataset: %s", exc)

        data = pd.DataFrame()
        tfidf_matrix = None
        tfidf_vectorizer = None


load_movie_dataset()


def find_local_match(title):
    """
    NEW: find a movie in movies.csv by title.

    Order: exact title -> title starts with the query -> title contains
    the query. Within a tier the movie with the most votes wins.
    Returns a DataFrame row index (label) or None.
    """
    if data.empty:
        return None

    query_key = canonical_title(title)

    if not query_key:
        return None

    keys = data["title_key"]
    votes = data["_votes"].fillna(0)

    for mask in (
        keys == query_key,
        keys.str.startswith(query_key, na=False),
        keys.str.contains(query_key, regex=False, na=False)
    ):
        candidates = data.index[mask]

        if len(candidates):
            return votes.loc[candidates].idxmax()

    return None


# ============================================================
# FORMAT LOCAL CSV MOVIE
# ============================================================

def format_local_movie(row, similarity=None):
    release_date = clean_str(row.get("release_date"))
    year = release_date[:4] if len(release_date) >= 4 else "N/A"

    poster_path = clean_str(row.get("poster_path")) or None
    rating = round(safe_float(row.get("vote_average")), 1)

    movie_id = safe_int(row.get("movie_id", row.get("id")))

    return {
        "id": movie_id,
        "source": "csv",
        "title": clean_str(row.get("title")),
        "year": year,
        "release_date": release_date,
        "rating": rating,
        "vote_average": rating,
        "genres": extract_genres(row.get("genres", "")),
        "overview": clean_str(row.get("overview")),
        "poster": tmdb_image(poster_path),
        "poster_path": poster_path,
        "backdrop": None,
        "similarity": (
            round(similarity * 100, 1)
            if similarity is not None
            else None
        )
    }


# ============================================================
# TMDB MOVIE (ONE CALL SERVES DETAILS, PROFILE AND TRAILER)
# ============================================================

def get_movie_full(movie_id):
    """
    Single cached request containing details, credits, keywords
    and videos. Details modal, recommendation profile, poster
    hydration and trailers all share this one cache entry.
    """
    return tmdb_request(
        f"movie/{movie_id}",
        params={
            "language": "en-US",
            "append_to_response": "credits,keywords,videos"
        }
    )


def full_matches_title(full, title):
    """Guard against a mismatched ID (e.g. CSV IDs that are not TMDB IDs)."""
    if not full or not title:
        return bool(full)

    return canonical_title(title) in title_keys(full)


def build_tmdb_feature_text(movie_details):
    """Build the query profile from TMDB metadata."""
    if not movie_details:
        return ""

    genres = [
        item.get("name", "")
        for item in movie_details.get("genres", []) or []
        if isinstance(item, dict)
    ]

    keyword_data = movie_details.get("keywords", {})
    keywords = []

    if isinstance(keyword_data, dict):
        keywords = [
            item.get("name", "")
            for item in (
                keyword_data.get("keywords")
                or keyword_data.get("results")
                or []
            )
            if isinstance(item, dict)
        ]

    credits = movie_details.get("credits", {}) or {}

    cast = [
        person.get("name", "")
        for person in credits.get("cast", [])[:CAST_LIMIT]
        if isinstance(person, dict)
    ]

    directors = [
        person.get("name", "")
        for person in credits.get("crew", [])
        if isinstance(person, dict)
        and person.get("job") in IMPORTANT_JOBS
    ]

    return build_feature_text(
        genres=genres,
        keywords=keywords,
        tagline=movie_details.get("tagline"),
        overview=movie_details.get("overview"),
        cast=cast,
        directors=directors
    )


# ============================================================
# CSV CONTENT-BASED RECOMMENDATIONS
# ============================================================

def csv_recommendations(
    title,
    limit=12,
    query_features=None,
    exclude_movie_id=None
):
    """
    Content-based recommender using TF-IDF and cosine similarity.

    If TMDB metadata is supplied, the searched movie does not need to
    exist in movies.csv. Otherwise the closest local title match is used
    (exact, then starts-with, then contains).
    """
    if data.empty or tfidf_matrix is None or tfidf_vectorizer is None:
        logger.warning("Local TF-IDF recommendations are unavailable.")
        return []

    title_key = canonical_title(title)

    if not title_key and not query_features:
        return []

    query_vector = None
    local_index = None

    if query_features and str(query_features).strip():
        candidate_vector = tfidf_vectorizer.transform([query_features])

        if candidate_vector.nnz:
            query_vector = candidate_vector

    if query_vector is None:
        local_index = find_local_match(title)

        if local_index is None:
            logger.info(
                "'%s' is not in movies.csv and has no usable TMDB profile.",
                title
            )
            return []

        query_vector = tfidf_matrix[local_index]

    scores = cosine_similarity(query_vector, tfidf_matrix).ravel()

    excluded = set(data.index[data["title_key"] == title_key].tolist())

    if local_index is not None:
        excluded.add(local_index)

    if exclude_movie_id is not None:
        excluded.update(
            data.index[data["_id"] == float(exclude_movie_id)].tolist()
        )

    recommendations = []

    for index in scores.argsort()[::-1]:
        score = float(scores[index])

        if score < MIN_LOCAL_SIMILARITY:
            break

        if index in excluded:
            continue

        votes = data.at[index, "_votes"]

        if pd.notna(votes) and votes < MIN_LOCAL_VOTES:
            continue

        movie = format_local_movie(data.iloc[index], similarity=score)

        if not movie["title"]:
            continue

        recommendations.append(movie)

        if len(recommendations) >= limit:
            break

    logger.info(
        "TF-IDF generated %s local recommendations for '%s'.",
        len(recommendations),
        title
    )

    return recommendations


# ============================================================
# TMDB TITLE SEARCH AND RANKING
# ============================================================

def search_rank(query_key, movie):
    """
    Sort key for search results:
      tier 0: exact title ("Avengers" -> "The Avengers")
      tier 1: title starts with the query ("Avengers: Endgame")
      tier 2: title contains the query
      tier 3: everything else
    Within a tier, the movie with more votes wins, which puts the
    famous film ahead of obscure namesakes.
    """
    keys = title_keys(movie)

    if query_key in keys:
        tier = 0
    elif any(key.startswith(query_key) for key in keys):
        tier = 1
    elif any(query_key in key for key in keys):
        tier = 2
    else:
        tier = 3

    return (
        tier,
        -float(movie.get("vote_count") or 0),
        -float(movie.get("popularity") or 0)
    )


def search_tmdb_movies(query):
    """Ranked raw TMDB search results (shared cache with autocomplete)."""
    query = " ".join(str(query or "").split()).lower()

    if not query or not tmdb_enabled():
        return []

    response = tmdb_request(
        "search/movie",
        params={
            "query": query,
            "include_adult": "false",
            "language": "en-US",
            "page": 1
        }
    )

    results = (response or {}).get("results", [])
    query_key = canonical_title(query)

    return sorted(results, key=lambda m: search_rank(query_key, m))


def find_best_tmdb_match(title):
    """
    Best TMDB match for a typed title.

    Prefers an exact title (ignoring a leading article), choosing the
    most-voted one. Falls back to a close fuzzy match for small typos,
    but never to a loosely related title.
    """
    results = search_tmdb_movies(title)

    if not results:
        return None

    query_key = canonical_title(title)

    if not query_key:
        return None

    exact = [m for m in results if query_key in title_keys(m)]

    if exact:
        selected = exact[0]
    else:
        selected, best_ratio = None, 0.0

        for movie in results[:10]:
            for key in title_keys(movie):
                ratio = SequenceMatcher(None, query_key, key).ratio()

                if ratio > best_ratio:
                    selected, best_ratio = movie, ratio

        if selected is None or best_ratio < FUZZY_THRESHOLD:
            logger.info(
                "No confident TMDB match for '%s'. Candidates: %s",
                title,
                [m.get("title") for m in results[:5]]
            )
            return None

    logger.info(
        "TMDB title match: '%s' -> '%s' (ID: %s)",
        title, selected.get("title"), selected.get("id")
    )

    return format_movie(selected)


def suggest_titles(title, count=3):
    """Titles for a 'Did you mean' message."""
    suggestions = []

    for movie in search_tmdb_movies(title)[:count]:
        year = (movie.get("release_date") or "")[:4]
        name = movie.get("title") or movie.get("original_title")

        if name:
            suggestions.append(f"{name} ({year})" if year else name)

    return suggestions


# ============================================================
# TMDB RECOMMENDATIONS
# ============================================================

def _tmdb_movie_list(endpoint, limit):
    response = tmdb_request(
        endpoint,
        params={"language": "en-US", "page": 1}
    )

    if not response:
        return []

    return [
        format_movie(movie)
        for movie in response.get("results", [])[:limit]
        if movie.get("id")
    ]


def get_tmdb_recommendations(movie_id, limit=12):
    return _tmdb_movie_list(f"movie/{movie_id}/recommendations", limit)


def get_tmdb_similar_movies(movie_id, limit=12):
    return _tmdb_movie_list(f"movie/{movie_id}/similar", limit)


def get_collection_movies(collection_id, exclude_id=None):
    """Other movies in the same franchise (sequels, prequels, ...)."""
    response = tmdb_request(
        f"collection/{collection_id}",
        params={"language": "en-US"}
    )

    parts = (response or {}).get("parts", [])

    return [
        format_movie(part)
        for part in parts
        if part.get("id") and part.get("id") != exclude_id
    ]


def pick_franchise(parts, reference_year, limit=MAX_FRANCHISE):
    """Keep the entries closest in time to the searched movie."""

    def year_of(movie):
        year = str(movie.get("year", ""))
        return int(year) if year.isdigit() else None

    if reference_year is not None and len(parts) > limit:
        parts = sorted(
            parts,
            key=lambda m: (
                abs(year_of(m) - reference_year)
                if year_of(m) is not None else 999
            )
        )[:limit]

    return sorted(parts, key=lambda m: m.get("release_date") or "9999")[:limit]


def discover_related_movies(movie_id, credits, limit=12):
    """Cast-based fallback using already-fetched credits."""
    cast_ids = [
        str(person["id"])
        for person in (credits or {}).get("cast", [])[:5]
        if person.get("id")
    ]

    if not cast_ids:
        return []

    response = tmdb_request(
        "discover/movie",
        params={
            "language": "en-US",
            "sort_by": "popularity.desc",
            "include_adult": "false",
            "include_video": "false",
            "vote_count.gte": 20,
            "with_cast": "|".join(cast_ids),
            "page": 1
        }
    )

    if not response:
        return []

    results = []

    for movie in response.get("results", []):
        if movie.get("id") == movie_id:
            continue

        formatted = format_movie(movie)

        if formatted:
            results.append(formatted)

        if len(results) >= limit:
            break

    return results


def discover_by_genre(profile, limit=24):
    """
    Same-genre fallback from TMDB. First movies that share the two main
    genres (AND), then movies that share any genre (OR). Popular,
    well-voted movies come first.
    """
    genre_ids = [
        str(genre["id"])
        for genre in (profile or {}).get("genres", []) or []
        if isinstance(genre, dict) and genre.get("id")
    ]

    if not genre_ids:
        return []

    results = []

    for separator, ids in ((",", genre_ids[:2]), ("|", genre_ids)):
        response = tmdb_request(
            "discover/movie",
            params={
                "language": "en-US",
                "sort_by": "vote_count.desc",
                "include_adult": "false",
                "include_video": "false",
                "vote_count.gte": 500,
                "with_genres": separator.join(ids),
                "page": 1
            }
        )

        for movie in (response or {}).get("results", []):
            formatted = format_movie(movie)

            if formatted:
                results.append(formatted)

        if len(results) >= limit:
            break

    return results[:limit]


def csv_genre_recommendations(profile, exclude_movie_id, title, limit=24):
    """
    Same-genre fallback from movies.csv: ranks local movies by genre
    overlap (Jaccard), then by vote count. Does not need TF-IDF.
    """
    if data.empty or "_genre_set" not in data.columns:
        return []

    target = {
        normalize_title(genre.get("name", ""))
        for genre in (profile or {}).get("genres", []) or []
        if isinstance(genre, dict)
    }
    target.discard("")

    if not target:
        return []

    title_key = canonical_title(title)
    scored = []

    for index, genre_set in data["_genre_set"].items():
        if not genre_set:
            continue

        overlap = len(target & genre_set)

        if not overlap:
            continue

        if data.at[index, "title_key"] == title_key:
            continue

        if (
            exclude_movie_id is not None
            and data.at[index, "_id"] == float(exclude_movie_id)
        ):
            continue

        votes = data.at[index, "_votes"]
        votes = 0 if pd.isna(votes) else votes

        if votes < MIN_LOCAL_VOTES:
            continue

        scored.append((overlap / len(target | genre_set), votes, index))

    scored.sort(reverse=True)

    return [
        format_local_movie(data.loc[index])
        for _, _, index in scored[:limit]
    ]


# ============================================================
# MERGING, DE-DUPLICATION, POSTERS
# ============================================================

def remove_duplicate_movies(movies, limit=12):
    unique = []
    seen_ids = set()
    seen_keys = set()

    for movie in movies:
        if not movie:
            continue

        movie_id = movie.get("id")
        key = movie_key(movie)

        if movie_id and movie_id in seen_ids:
            continue

        if key[0] and key in seen_keys:
            continue

        if movie_id:
            seen_ids.add(movie_id)

        seen_keys.add(key)
        unique.append(movie)

        if len(unique) >= limit:
            break

    return unique


def interleave(*lists):
    """Alternate items from several lists: a1, b1, a2, b2, ..."""
    merged = []
    longest = max((len(items) for items in lists), default=0)

    for position in range(longest):
        for items in lists:
            if position < len(items):
                merged.append(items[position])

    return merged


def finalize_results(movies, matched, limit):
    searched_id = matched.get("id")
    searched_key = movie_key(matched)

    filtered = [
        movie for movie in movies
        if movie
        and not (
            searched_id is not None
            and str(movie.get("id")) == str(searched_id)
        )
        and movie_key(movie) != searched_key
    ]

    return remove_duplicate_movies(filtered, limit=limit)


def hydrate_local_posters(movies):
    """
    CSV rows often have no poster. Fill them from the shared cached
    TMDB movie call, but only for the movies actually displayed and
    only when the TMDB title matches (protects against wrong IDs).
    """
    if not HYDRATE_LOCAL_POSTERS or not tmdb_enabled():
        return movies

    targets = [
        movie for movie in movies
        if movie.get("source") == "csv"
        and not movie.get("poster")
        and movie.get("id")
    ]

    if not targets:
        return movies

    def hydrate(movie):
        full = get_movie_full(movie["id"])

        if not full or canonical_title(movie["title"]) not in title_keys(full):
            return

        official = format_movie(full)

        movie["poster"] = official["poster"]
        movie["poster_path"] = official["poster_path"]
        movie["backdrop"] = official["backdrop"]

        if not movie.get("rating"):
            movie["rating"] = official["rating"]
            movie["vote_average"] = official["vote_average"]

        if movie.get("year") == "N/A":
            movie["year"] = official["year"]

        if not movie.get("overview"):
            movie["overview"] = official["overview"]

    with ThreadPoolExecutor(max_workers=6) as executor:
        list(executor.map(hydrate, targets))

    return movies


# ============================================================
# MAIN RECOMMENDATION FUNCTION
# ============================================================

def recommend_for_tmdb_movie(matched, profile, limit=12):
    movie_id = matched["id"]

    collection = (profile or {}).get("belongs_to_collection") or {}
    collection_id = collection.get("id")

    year = str(matched.get("year", ""))
    reference_year = int(year) if year.isdigit() else None

    # TMDB calls run in the background while TF-IDF is computed here.
    with ThreadPoolExecutor(max_workers=2) as executor:
        recs_future = executor.submit(
            get_tmdb_recommendations, movie_id, limit * 2
        )

        franchise_future = (
            executor.submit(get_collection_movies, collection_id, movie_id)
            if collection_id else None
        )

        local_results = csv_recommendations(
            title=matched["title"],
            limit=limit * 2,
            query_features=build_tmdb_feature_text(profile),
            exclude_movie_id=movie_id
        )

        tmdb_results = recs_future.result()
        franchise = (
            franchise_future.result() if franchise_future else []
        )

    franchise = pick_franchise(franchise, reference_year)

    logger.info(
        "Franchise: %s, TMDB recs: %s, TF-IDF: %s",
        len(franchise), len(tmdb_results), len(local_results)
    )

    # Order: same franchise first, then TMDB and TF-IDF alternating.
    combined = franchise + interleave(tmdb_results, local_results)
    results = finalize_results(combined, matched, limit)

    # Optional extra calls, only when the list is still short.
    if len(results) < limit:
        combined += get_tmdb_similar_movies(movie_id, (limit - len(results)) * 2)
        results = finalize_results(combined, matched, limit)

    if len(results) < limit:
        combined += discover_related_movies(
            movie_id,
            (profile or {}).get("credits"),
            (limit - len(results)) * 2
        )
        results = finalize_results(combined, matched, limit)

    # Guaranteed same-genre top-up (TMDB discover, then local CSV).
    if len(results) < limit:
        combined += discover_by_genre(profile, limit * 2)
        results = finalize_results(combined, matched, limit)

    if len(results) < limit:
        combined += csv_genre_recommendations(
            profile, movie_id, matched["title"], limit * 2
        )
        results = finalize_results(combined, matched, limit)

    logger.info(
        "Final recommendations for '%s': %s",
        matched.get("title"), len(results)
    )

    return hydrate_local_posters(results)


def recommend_movies(title, movie_id=None, limit=12):
    """
    Hybrid recommendation strategy:

    1. Identify the searched movie (by TMDB ID when the user picked an
       autocomplete suggestion, otherwise by ranked title search, and
       finally through the local movies.csv dataset).
    2. Show movies from the same franchise first.
    3. Blend TMDB recommendations with local TF-IDF + cosine results.
    4. Top up with similar / cast-based movies only if needed.
    5. De-duplicate and exclude the searched movie.
    """
    title = (title or "").strip()

    if not title and not movie_id:
        return None, []

    matched = None
    profile = None

    if movie_id:
        full = get_movie_full(movie_id)

        if full and full_matches_title(full, title):
            matched = format_movie(full)
            profile = full

    if not matched and title:
        matched = find_best_tmdb_match(title)

        if matched and matched.get("id"):
            profile = get_movie_full(matched["id"])

    # NEW: TMDB search did not find it, but the title may exist in
    # movies.csv. Use its ID to fetch the real TMDB profile.
    if not (matched and matched.get("id")) and title:
        local_index = find_local_match(title)

        if local_index is not None:
            local_id = safe_int(data.at[local_index, "_id"])
            local_title = data.at[local_index, "title"]

            if local_id:
                full = get_movie_full(local_id)

                if full and full_matches_title(full, local_title):
                    matched = format_movie(full)
                    profile = full
                    logger.info(
                        "Matched '%s' through movies.csv -> '%s' (ID: %s)",
                        title, local_title, local_id
                    )

    if matched and matched.get("id"):
        logger.info(
            "Matched searched movie: %s (ID: %s)",
            matched.get("title"), matched.get("id")
        )
        return matched, recommend_for_tmdb_movie(matched, profile, limit)

    # TMDB is unavailable or did not identify the movie.
    logger.info("TMDB did not identify '%s'. Using movies.csv only.", title)

    local_results = csv_recommendations(title=title, limit=limit)

    if not local_results:
        return None, []

    local_index = find_local_match(title)

    if local_index is not None:
        matched_local = format_local_movie(data.loc[local_index])
    else:
        matched_local = {"id": None, "title": title, "year": "N/A"}

    return matched_local, local_results


# ============================================================
# MOVIE DETAILS
# ============================================================

def get_movie_details(movie_id, expected_title=None):
    full = get_movie_full(movie_id)

    if not full or not full_matches_title(full, expected_title):
        return None

    movie = format_movie(full)
    credits = full.get("credits", {}) or {}

    cast_names = [
        person.get("name")
        for person in credits.get("cast", [])[:10]
        if person.get("name")
    ]

    directors = [
        person.get("name")
        for person in credits.get("crew", [])
        if person.get("job") == "Director"
    ]

    movie["cast"] = ", ".join(cast_names) or "Unavailable"
    movie["director"] = ", ".join(directors) or "Unavailable"
    movie["runtime"] = full.get("runtime")
    movie["tagline"] = full.get("tagline")
    movie["status"] = full.get("status")
    movie["budget"] = full.get("budget")
    movie["revenue"] = full.get("revenue")

    return movie


# ============================================================
# TRAILER SEARCH
# ============================================================

def pick_trailer(videos_block):
    videos = (videos_block or {}).get("results", [])

    trailers = [
        video for video in videos
        if (
            video.get("site") == "YouTube"
            and video.get("type") == "Trailer"
            and video.get("key")
        )
    ]

    if not trailers:
        return None

    official = [video for video in trailers if video.get("official")]
    selected = (official or trailers)[0]

    return {
        "key": selected["key"],
        "name": selected.get("name"),
        "url": "https://www.youtube.com/watch?v=" + selected["key"]
    }


def search_youtube_trailer(title):
    """Last resort (costs YouTube quota), so results are cached."""
    if not YOUTUBE_API_KEY:
        return None

    cache_key = f"youtube:{canonical_title(title)}"
    cached = get_cache(cache_key)

    if cached is not _MISS:
        return cached

    result = None

    try:
        response = session.get(
            "https://www.googleapis.com/youtube/v3/search",
            params={
                "part": "snippet",
                "q": f"{title} official movie trailer",
                "type": "video",
                "maxResults": 5,
                "key": YOUTUBE_API_KEY
            },
            timeout=REQUEST_TIMEOUT
        )

        response.raise_for_status()

        items = response.json().get("items", [])
        video_id = items[0].get("id", {}).get("videoId") if items else None

        if video_id:
            result = {
                "key": video_id,
                "name": title,
                "url": f"https://www.youtube.com/watch?v={video_id}"
            }

    except (requests.exceptions.RequestException, ValueError) as exc:
        logger.error("YouTube API request failed: %s", exc)
        set_cache(cache_key, None, FAILURE_CACHE_DURATION)
        return None

    set_cache(cache_key, result, CACHE_DURATION)
    return result


def trailer_for(movie_id=None, title=""):
    """
    Trailer lookup. A known TMDB ID is used directly (no ambiguous
    title search, and no extra request: it reuses the cached movie
    call). The title search and YouTube are only fallbacks.
    """
    title = (title or "").strip()

    if movie_id:
        full = get_movie_full(movie_id)

        if full and full_matches_title(full, title):
            trailer = pick_trailer(full.get("videos"))

            if trailer:
                return trailer

    if title:
        match = find_best_tmdb_match(title)

        if match and match.get("id") and match["id"] != movie_id:
            trailer = pick_trailer(
                (get_movie_full(match["id"]) or {}).get("videos")
            )

            if trailer:
                return trailer

        return search_youtube_trailer(title)

    return None


# ============================================================
# RENDER HOME
# ============================================================

def render_home(
    recommendations=None,
    searched_title="",
    error=None,
    home=None,
    searched_movie=None
):
    if home is None:
        home = get_home_data()

    return render_template(
        "index.html",
        home=home,
        recommendations=recommendations or [],
        searched_title=searched_title,
        searched_movie=searched_movie,   # NEW: the movie that was searched
        error=error,
        tmdb_enabled=tmdb_enabled()
    )


# ============================================================
# FLASK ROUTES
# ============================================================

@app.route("/", methods=["GET"])
def index():
    return render_home()


@app.route("/recommend", methods=["GET", "POST"])
def recommend():
    # request.values covers both the query string and form body.
    title = (
        request.values.get("movie")
        or request.values.get("title")
        or request.values.get("search")
        or ""
    ).strip()

    movie_id = safe_int(request.values.get("movie_id"))

    if not title and not movie_id:
        return render_home(error="Please enter a movie title.")

    matched_movie, recommendations = recommend_movies(
        title,
        movie_id=movie_id
    )

    if not matched_movie:
        if not tmdb_enabled():
            message = (
                "TMDB is not configured. Please check your "
                ".env file and API credentials."
            )
        else:
            message = (
                f'No exact movie match was found for "{title}". '
                "Check the spelling or choose a movie from the "
                "suggestions."
            )

            suggestions = suggest_titles(title)

            if suggestions:
                message += " Did you mean: " + ", ".join(suggestions) + "?"

        return render_home(searched_title=title, error=message)

    searched_title = matched_movie.get("title") or title

    # The exact match always comes first, then the similar movies.
    exact_movie = dict(matched_movie)
    exact_movie["exact"] = True
    exact_movie["similarity"] = None

    if not recommendations:
        return render_home(
            recommendations=[exact_movie],
            searched_title=searched_title,
            searched_movie=matched_movie,
            error=(
                "No similar movies could be loaded "
                f'for "{searched_title}" right now. Please try again.'
            )
        )

    return render_home(
        recommendations=[exact_movie] + recommendations,
        searched_title=searched_title,
        searched_movie=matched_movie
    )


@app.route("/api/search", methods=["GET"])
def api_search():
    """Returns a JSON array of ranked autocomplete suggestions."""
    query = request.args.get("q", "").strip()

    if len(query) < 2:
        return jsonify([])

    results = []
    seen = set()

    for movie in search_tmdb_movies(query)[:10]:
        formatted = format_movie(movie)

        if not formatted:
            continue

        key = movie_key(formatted)

        if key in seen:
            continue

        seen.add(key)
        results.append(formatted)

    # The local CSV only fills in when TMDB gave few suggestions
    # (for example when TMDB is unreachable).
    if len(results) < 5 and not data.empty:
        query_key = canonical_title(query)

        matches = data[
            data["title"].str.contains(
                query, case=False, regex=False, na=False
            )
        ].copy()

        matches["_starts"] = ~matches["title_key"].str.startswith(query_key)
        matches = matches.sort_values(["_starts", "_votes"],
                                      ascending=[True, False]).head(8)

        for _, row in matches.iterrows():
            formatted = format_local_movie(row)
            key = movie_key(formatted)

            if key in seen:
                continue

            seen.add(key)
            results.append(formatted)

    return jsonify(results[:10])


@app.route("/api/debug/<path:title>", methods=["GET"])
def api_debug(title):
    """
    NEW: diagnose a search, e.g. /api/debug/interstellar
    Shows what TMDB matched, what the CSV matched, and how many
    recommendations each source produced.
    """
    info = {
        "query": title,
        "tmdb_enabled": tmdb_enabled(),
        "csv_rows": int(len(data)),
    }

    tmdb_match = find_best_tmdb_match(title) if tmdb_enabled() else None
    info["tmdb_match"] = (
        {k: tmdb_match[k] for k in ("id", "title", "year")}
        if tmdb_match else None
    )

    local_index = find_local_match(title)
    info["csv_match"] = (
        {
            "title": data.at[local_index, "title"],
            "id": safe_int(data.at[local_index, "_id"]),
            "votes": safe_int(data.at[local_index, "_votes"]),
        }
        if local_index is not None else None
    )

    if tmdb_match and tmdb_match.get("id"):
        mid = tmdb_match["id"]
        profile = get_movie_full(mid)
        info["profile_loaded"] = bool(profile)
        info["tmdb_recommendations"] = len(get_tmdb_recommendations(mid, 24))
        info["genre_discover"] = len(discover_by_genre(profile, 24))
        info["csv_genre"] = len(
            csv_genre_recommendations(profile, mid, tmdb_match["title"], 24)
        )
        info["tfidf_recommendations"] = len(
            csv_recommendations(
                tmdb_match["title"],
                limit=24,
                query_features=build_tmdb_feature_text(profile),
                exclude_movie_id=mid
            )
        )

    return jsonify(info)


@app.route("/api/home", methods=["GET"])
def api_home():
    force_refresh = (
        request.args.get("refresh", "").lower() in ["1", "true", "yes"]
    )

    return jsonify(get_home_data(force_refresh=force_refresh))


@app.route("/api/movie/<int:movie_id>", methods=["GET"])
def api_movie(movie_id):
    movie = get_movie_details(
        movie_id,
        expected_title=request.args.get("title", "").strip()
    )

    if not movie:
        return jsonify({"error": "Movie details could not be found."}), 404

    return jsonify(movie)


@app.route("/api/trailer/<int:movie_id>", methods=["GET"])
def api_trailer(movie_id):
    trailer = trailer_for(movie_id=movie_id)

    if not trailer:
        return jsonify({"error": "Trailer not found."}), 404

    return jsonify(trailer)


@app.route("/api/trailer/search", methods=["GET"])
def api_trailer_search():
    title = request.args.get("title", "").strip()
    movie_id = safe_int(request.args.get("id"))

    if not title and not movie_id:
        return jsonify({"error": "Movie title is required."}), 400

    trailer = trailer_for(movie_id=movie_id, title=title)

    if not trailer:
        return jsonify({"error": "Trailer could not be found."}), 404

    return jsonify(trailer)


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(404)
def page_not_found(error):
    if request.path.startswith("/api/"):
        return jsonify({"error": "API endpoint not found."}), 404

    return render_home(error="The requested page could not be found."), 404


@app.errorhandler(500)
def internal_server_error(error):
    logger.exception("Internal server error: %s", error)

    if request.path.startswith("/api/"):
        return jsonify({"error": "An internal server error occurred."}), 500

    return render_home(error="Something went wrong. Please try again."), 500


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=int(os.getenv("PORT", 5000)),
        debug=os.getenv("FLASK_DEBUG", "1") == "1"
    )