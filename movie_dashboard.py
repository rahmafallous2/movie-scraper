import streamlit as st
import pandas as pd
import numpy as np
import difflib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
import re
from datetime import datetime

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="🎬 CineAI: Smart Movie Analytics & Recommendations",
    layout="wide",
    page_icon="🎬",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1E3A8A;
    text-align: center;
    margin-bottom: 1rem;
}
.feature-card {
    background-color: #f8f9fa;
    padding: 1.5rem;
    border-radius: 10px;
    border-left: 5px solid #1E3A8A;
    margin-bottom: 1rem;
}
.highlight {
    background-color: #E3F2FD;
    padding: 0.5rem;
    border-radius: 5px;
    border-left: 3px solid #2196F3;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🎬 CineAI: Smart Movie Analytics & Recommendations</h1>', unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        file_paths = [
            'FinalCleaned_data.csv'
        ]
        df = None
        for path in file_paths:
            try:
                df = pd.read_csv(path)
                break
            except Exception:
                continue
        
        if df is None:
            st.error("Could not find data file. Please upload your CSV.")
            uploaded_file = st.file_uploader("Upload your movie data CSV", type=['csv'])
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
            else:
                return pd.DataFrame()
        
        if 'index' not in df.columns:
            df.reset_index(inplace=True)
            df.rename(columns={'index': 'original_index'}, inplace=True)
            df['index'] = range(len(df))
        
        if 'vote_average' in df.columns:
            rating_mapping = {
                'Low': 2.0,
                'Medium': 5.0,
                'High': 8.0,
                'Unknown': 0.0
            }
            df['vote_average'] = df['vote_average'].apply(
                lambda x: rating_mapping.get(x, x) if isinstance(x, str) else x
            )
            df['vote_average'] = pd.to_numeric(df['vote_average'], errors='coerce')
            df['vote_average'].fillna(0, inplace=True)
        
        if 'vote_average' in df.columns:
            if df['vote_average'].max() > 10 or df['vote_average'].min() < 0:
                current_min = df['vote_average'].min()
                current_max = df['vote_average'].max()
                if current_max > current_min:
                    df['vote_average'] = (
                        (df['vote_average'] - current_min) /
                        (current_max - current_min) * 10
                    )
                else:
                    df['vote_average'] = 5.0
            elif (df['vote_average'].max() <= 1.0 and 
                df['vote_average'].min() >= -1.0 and 
                df['vote_average'].max() > 0):
                df['vote_average'] = df['vote_average'] * 10
        
        if 'popularity' in df.columns:
            df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce').fillna(0)
            if df['popularity'].max() > 1000:
                df['popularity_log'] = np.log1p(df['popularity'])
                pop_99th = df['popularity'].quantile(0.99)
                df['popularity_capped'] = np.where(
                    df['popularity'] > pop_99th,
                    pop_99th,
                    df['popularity']
                )
        
        text_columns = ['genres', 'keywords', 'tagline', 'overview', 'cast', 'director']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].fillna('')
            else:
                df[col] = ''
        
        if 'release_date' in df.columns:
            df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
            df['release_year'] = df['release_date'].dt.year
        
        numeric_columns = ['budget', 'revenue', 'vote_average', 'vote_count', 'runtime']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        def _extract_release_year(movie):
            if 'release_date' in movie and pd.notna(movie['release_date']):
                try:
                    date_obj = pd.to_datetime(movie['release_date'], errors='coerce')
                    if pd.notna(date_obj):
                        year = date_obj.year
                        if 1900 < year < 2030:
                            return year
                except Exception:
                    pass
            if 'title' in movie and pd.notna(movie['title']):
                title = str(movie['title'])
                year_match = re.search(r'\((\d{4})\)$', title)
                if year_match:
                    try:
                        year = int(year_match.group(1))
                        if 1900 < year < 2030:
                            return year
                    except Exception:
                        pass
            return None

        def fix_release_years(_df):
            fixed_count = 0
            if 'release_year' not in _df.columns:
                _df['release_year'] = np.nan
            for idx, movie in _df.iterrows():
                current_year = _df.at[idx, 'release_year']
                if pd.isna(current_year) or current_year == 0 or current_year == "N/A" or current_year == "NaN":
                    fixed_year = _extract_release_year(movie)
                    if fixed_year and 1900 < fixed_year < 2030:
                        _df.at[idx, 'release_year'] = fixed_year
                        fixed_count += 1
            if fixed_count > 0:
                print(f"Fixed {fixed_count} missing release years")
            return _df
        
        df = fix_release_years(df)
        
        if 'budget' in df.columns and 'revenue' in df.columns:
            df['budget'] = pd.to_numeric(df['budget'], errors='coerce').fillna(0)
            df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce').fillna(0)
            df['profit'] = df['revenue'] - df['budget']
            df['roi'] = np.where(
                df['budget'] > 0,
                (df['profit'] / df['budget']) * 100,
                0
            )
            
            df['success_category'] = df['roi'].apply(
                lambda x: "Flop" if x < 0 else "Average" if 0 <= x < 50 else "Hit" if 50 <= x < 200 else "Super Hit"
            )
        
        def detect_movie_mood(overview):
            if not overview or overview == 'Unknown' or overview == 'nan':
                return "Neutral"
            
            overview_lower = str(overview).lower()
            mood_keywords = {
                'heartwarming': ['love', 'family', 'friendship', 'heart', 'care', 'bond', 'together'],
                'exciting': ['action', 'adventure', 'thriller', 'chase', 'battle', 'fight', 'mission'],
                'emotional': ['drama', 'emotional', 'heartbreak', 'loss', 'struggle', 'pain'],
                'mysterious': ['mystery', 'secret', 'unknown', 'investigate', 'solve', 'clue'],
                'epic': ['epic', 'journey', 'quest', 'legend', 'kingdom', 'ancient', 'destiny'],
                'funny': ['comedy', 'funny', 'hilarious', 'laugh', 'humor', 'joke', 'amusing'],
                'dark': ['dark', 'horror', 'terror', 'fear', 'death', 'brutal', 'violent']
            }
            
            mood_scores = {}
            for mood, keywords in mood_keywords.items():
                score = sum(3 for keyword in keywords if keyword in overview_lower)
                mood_scores[mood] = score
            
            if mood_scores:
                dominant_mood = max(mood_scores.items(), key=lambda x: x[1])
                return dominant_mood[0] if dominant_mood[1] > 0 else "Neutral"
            
            return "Neutral"
        
        if 'overview' in df.columns:
            df['mood'] = df['overview'].apply(detect_movie_mood)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.error("❌ No data loaded. Please check your CSV file.")
    st.stop()

with st.expander("📁 Dataset Information", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Movies", len(df))
    with col2:
        st.metric("Columns", len(df.columns))
    with col3:
        years = "N/A"
        if 'release_year' in df.columns and not df['release_year'].isna().all():
            min_year = int(df['release_year'].min())
            max_year = int(df['release_year'].max())
            years = f"{min_year}-{max_year}"
        st.metric("Year Range", years)
    
    sample_cols = [c for c in ['title', 'release_year', 'director', 'vote_average', 'genres'] if c in df.columns]
    if sample_cols:
        st.write("**Sample Data:**")
        st.dataframe(df[sample_cols].head(10), use_container_width=True)

class SmartMovieSearch:
    def __init__(self, movies_data):
        self.movies_data = movies_data
        self._build_search_index()
    
    def _build_search_index(self):
        self.movies_data['search_text'] = (
            self.movies_data['title'].fillna('') + ' ' +
            self.movies_data['director'].fillna('') + ' ' +
            self.movies_data['cast'].fillna('') + ' ' +
            self.movies_data['genres'].fillna('') + ' ' +
            self.movies_data['keywords'].fillna('') + ' ' +
            self.movies_data['overview'].fillna('')
        ).str.lower()
        self._build_autocomplete_data()
    
    def _build_autocomplete_data(self):
        self.titles = self.movies_data['title'].dropna().unique().tolist()
        
        all_directors = []
        for directors in self.movies_data['director'].dropna():
            if isinstance(directors, str):
                director_list = [d.strip() for d in directors.split(',')]
                all_directors.extend(director_list)
        self.directors = list(set(all_directors))
        
        all_genres = []
        for genres in self.movies_data['genres'].dropna():
            if isinstance(genres, str):
                clean_genres = genres.replace('[', '').replace(']', '').replace('{', '').replace('}', '').replace('"', '')
                genre_list = [genre.strip() for genre in clean_genres.split(',') if genre.strip()]
                all_genres.extend(genre_list)
        self.genres = list(set(all_genres))
        
        all_cast = []
        for cast in self.movies_data['cast'].dropna():
            if isinstance(cast, str):
                cast_list = [c.strip() for c in cast.split(',')[:3]]
                all_cast.extend(cast_list)
        self.cast_members = list(set(all_cast))
    
    def smart_search(self, query, filters=None, max_results=20):
        if not query and not filters:
            return self._get_popular_movies(max_results)
        
        results = []
        
        if query:
            text_results = self._text_search(query, max_results * 2)
            results.extend(text_results)
        
        if filters:
            if results:
                filtered_results = self._apply_filters(results, filters)
            else:
                filtered_results = self._apply_filters(
                    [{'movie': movie, 'score': 1.0, 'match_type': 'filter'} 
                    for _, movie in self.movies_data.iterrows()],
                    filters
                )
            results = filtered_results
        
        unique_results = self._deduplicate_results(results)
        sorted_results = sorted(unique_results, key=lambda x: x['score'], reverse=True)
        return sorted_results[:max_results]
    
    def _text_search(self, query, max_results):
        query = query.lower().strip()
        results = []
        
        exact_title_matches = self.movies_data[
            self.movies_data['title'].str.lower() == query
        ]
        for _, movie in exact_title_matches.iterrows():
            results.append({
                'movie': movie,
                'score': 1.0,
                'match_type': 'exact_title',
                'matched_field': 'title'
            })
        
        partial_title_matches = self.movies_data[
            self.movies_data['title'].str.lower().str.contains(query, na=False)
        ]
        for _, movie in partial_title_matches.iterrows():
            try:
                title_len = max(len(str(movie['title'])), 1)
                score = 0.8 + (0.1 * (len(query) / title_len))
            except Exception:
                score = 0.85
            results.append({
                'movie': movie,
                'score': min(score, 0.95),
                'match_type': 'partial_title',
                'matched_field': 'title'
            })
        
        title_matches = difflib.get_close_matches(query, self.titles, n=10, cutoff=0.6)
        for title in title_matches:
            movie = self.movies_data[self.movies_data['title'] == title].iloc[0]
            results.append({
                'movie': movie,
                'score': 0.7,
                'match_type': 'fuzzy_title',
                'matched_field': 'title'
            })
        
        director_matches = self.movies_data[
            self.movies_data['director'].str.lower().str.contains(query, na=False)
        ]
        for _, movie in director_matches.iterrows():
            results.append({
                'movie': movie,
                'score': 0.6,
                'match_type': 'director',
                'matched_field': 'director'
            })
        
        genre_matches = self.movies_data[
            self.movies_data['genres'].str.lower().str.contains(query, na=False)
        ]
        for _, movie in genre_matches.iterrows():
            results.append({
                'movie': movie,
                'score': 0.5,
                'match_type': 'genre',
                'matched_field': 'genres'
            })
        
        tfidf_matches = self._tfidf_search(query, max_results // 2)
        results.extend(tfidf_matches)
        
        return results
    
    def _tfidf_search(self, query, max_results):
        try:
            vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
            tfidf_matrix = vectorizer.fit_transform(self.movies_data['search_text'].fillna(''))
            query_vec = vectorizer.transform([query])
            similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
            top_indices = similarities.argsort()[-max_results:][::-1]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.1:
                    results.append({
                        'movie': self.movies_data.iloc[idx],
                        'score': float(similarities[idx]) * 0.3,
                        'match_type': 'content',
                        'matched_field': 'overview/keywords'
                    })
            return results
        except Exception:
            return []
    
    def _apply_filters(self, results, filters):
        filtered_results = []
        for result in results:
            movie = result['movie']
            passes_filters = True
            
            if 'year' in filters:
                year = self._get_release_year(movie)
                if year and year != "N/A":
                    try:
                        movie_year = int(year)
                        if movie_year != filters['year']:
                            passes_filters = False
                    except Exception:
                        passes_filters = False
            
            if 'min_rating' in filters:
                if movie.get('vote_average', 0) < filters['min_rating']:
                    passes_filters = False
            
            if 'genres' in filters and filters['genres']:
                genre_match = any(
                    genre.lower() in str(movie.get('genres', '')).lower()
                    for genre in filters['genres']
                )
                if not genre_match:
                    passes_filters = False
            
            if 'mood' in filters and filters['mood']:
                if movie.get('mood') != filters['mood']:
                    passes_filters = False
            
            if 'max_runtime' in filters:
                try:
                    runtime = float(movie.get('runtime', 0))
                except Exception:
                    runtime = 0
                if runtime > filters['max_runtime']:
                    passes_filters = False
            
            if 'success_level' in filters and filters['success_level']:
                if movie.get('success_category') != filters['success_level']:
                    passes_filters = False
            
            if passes_filters:
                filtered_results.append(result)
        
        return filtered_results
    
    def _deduplicate_results(self, results):
        seen_movies = set()
        unique_results = []
        for result in results:
            movie_title = result['movie']['title']
            if movie_title not in seen_movies:
                seen_movies.add(movie_title)
                unique_results.append(result)
            else:
                for existing in unique_results:
                    if existing['movie']['title'] == movie_title:
                        if result['score'] > existing['score']:
                            existing.update(result)
                        break
        return unique_results
    
    def _get_release_year(self, movie):
        if 'release_year' in movie and pd.notna(movie['release_year']):
            try:
                year = int(movie['release_year'])
                if 1900 < year < 2030:
                    return str(year)
            except Exception:
                pass
        
        if 'release_date' in movie and pd.notna(movie['release_date']):
            try:
                date_obj = pd.to_datetime(movie['release_date'], errors='coerce')
                if pd.notna(date_obj):
                    year = date_obj.year
                    if 1900 < year < 2030:
                        return str(year)
            except Exception:
                pass
        
        return "N/A"
    
    def _get_popular_movies(self, max_results):
        if 'popularity' in self.movies_data.columns:
            base = self.movies_data.copy()
            base['popularity'] = pd.to_numeric(base['popularity'], errors='coerce').fillna(0)
            popular_movies = base.nlargest(max_results, 'popularity')
        elif 'vote_average' in self.movies_data.columns:
            popular_movies = self.movies_data.nlargest(max_results, 'vote_average')
        else:
            popular_movies = self.movies_data.sample(
                min(max_results, len(self.movies_data)),
                random_state=42
            )
        
        return [{
            'movie': movie,
            'score': 0.5,
            'match_type': 'popular',
            'matched_field': 'popularity'
        } for _, movie in popular_movies.iterrows()]
    
    def get_autocomplete_suggestions(self, query, max_suggestions=5):
        query = query.lower().strip()
        suggestions = []
        
        title_matches = [title for title in self.titles if query in title.lower()]
        suggestions.extend([f"{title}" for title in title_matches[:max_suggestions]])
        
        director_matches = [director for director in self.directors if query in director.lower()]
        suggestions.extend([f"{director}" for director in director_matches[:max_suggestions]])
        
        genre_matches = [genre for genre in self.genres if query in genre.lower()]
        suggestions.extend([f"{genre}" for genre in genre_matches[:max_suggestions]])
        
        cast_matches = [cast for cast in self.cast_members if query in cast.lower()]
        suggestions.extend([f"{cast}" for cast in cast_matches[:max_suggestions]])
        
        return suggestions[:max_suggestions]

class MovieSkillBuilder:
    def __init__(self, movies_data):
        self.movies_data = movies_data
    
    def extract_skills_from_movie(self, movie_title):
        movie_data = self._find_movie_data(movie_title)
        if movie_data is None:
            return None, f"Movie '{movie_title}' not found."
        
        overview = str(movie_data.get('overview', ''))
        keywords = str(movie_data.get('keywords', ''))
        skills_report = []
        
        cooking_skills = self._analyze_cooking_skills(overview, keywords)
        science_concepts = self._analyze_science_concepts(overview, keywords)
        historical_facts = self._analyze_historical_facts(overview, keywords)
        survival_skills = self._analyze_survival_skills(overview, keywords)
        medical_scenes = self._analyze_medical_scenes(overview, keywords)
        technology_scenes = self._analyze_technology_scenes(overview, keywords)
        
        educational_value = self._calculate_educational_value(overview, keywords)
        
        skills_report.append(f"**SKILL ANALYSIS: {movie_data['title']}**")
        skills_report.append("="*50)
        skills_report.append(cooking_skills)
        skills_report.append(science_concepts)
        skills_report.append(historical_facts)
        skills_report.append(survival_skills)
        skills_report.append(medical_scenes)
        skills_report.append(technology_scenes)
        skills_report.append(educational_value)
        
        return "\n".join(skills_report), None
    
    def _analyze_cooking_skills(self, overview, keywords):
        cooking_keywords = {
            'cooking': ['cook', 'chef', 'recipe', 'kitchen', 'bake', 'food', 'culinary'],
            'baking': ['bake', 'pastry', 'bread', 'cake', 'oven', 'dessert'],
            'mixology': ['cocktail', 'bartender', 'drink', 'mix', 'bar'],
            'techniques': ['grill', 'fry', 'roast', 'saute', 'chop', 'slice']
        }
        
        found_skills = []
        overview_lower = overview.lower()
        for skill_type, words in cooking_keywords.items():
            if any(word in overview_lower for word in words):
                found_skills.append(skill_type)
        
        if found_skills:
            return f"\n**COOKING & FOOD SKILLS:**\nFound: {', '.join(found_skills)}\nLearn about: Food preparation, cooking techniques, culinary arts"
        else:
            return f"\n**COOKING & FOOD SKILLS:**\nNo significant cooking content found"
    
    def _analyze_science_concepts(self, overview, keywords):
        science_keywords = {
            'physics': ['physics', 'quantum', 'space', 'gravity', 'energy', 'experiment'],
            'biology': ['biology', 'dna', 'evolution', 'genetics', 'virus', 'disease'],
            'chemistry': ['chemistry', 'chemical', 'element', 'reaction', 'formula'],
            'astronomy': ['space', 'planet', 'star', 'galaxy', 'universe', 'astronaut'],
            'technology': ['technology', 'invention', 'robot', 'ai', 'computer', 'code']
        }
        
        found_concepts = []
        overview_lower = overview.lower()
        for concept, words in science_keywords.items():
            if any(word in overview_lower for word in words):
                found_concepts.append(concept)
        
        if found_concepts:
            return f"\n**SCIENCE & TECHNOLOGY CONCEPTS:**\nFound: {', '.join(found_concepts)}\nLearn about: Scientific principles, technological innovation"
        else:
            return f"\n**SCIENCE & TECHNOLOGY:**\nNo significant science content found"
    
    def _analyze_historical_facts(self, overview, keywords):
        history_keywords = {
            'ancient': ['ancient', 'rome', 'egypt', 'greek', 'medieval', 'kingdom'],
            'modern': ['world war', 'war', 'revolution', 'historical', 'century', 'era'],
            'biographical': ['biography', 'true story', 'based on true', 'real life'],
            'cultural': ['culture', 'tradition', 'heritage', 'historical period']
        }
        
        found_history = []
        overview_lower = overview.lower()
        for period, words in history_keywords.items():
            if any(word in overview_lower for word in words):
                found_history.append(period)
        
        if found_history:
            result = f"\n**HISTORICAL CONTEXT:**\nFound: {', '.join(found_history)}\nLearn about: Historical events, cultural contexts, real stories"
            if 'world war' in overview_lower:
                result += "\nSuggested topics: WWII history, military strategy, geopolitical impacts"
            if 'ancient' in overview_lower:
                result += "\nSuggested topics: Ancient civilizations, archaeology, historical artifacts"
            return result
        else:
            return f"\n**HISTORICAL CONTEXT:**\nNo significant historical content found"
    
    def _analyze_survival_skills(self, overview, keywords):
        survival_keywords = {
            'wilderness': ['wilderness', 'survival', 'island', 'jungle', 'forest', 'mountain'],
            'disaster': ['disaster', 'earthquake', 'storm', 'apocalypse', 'catastrophe'],
            'adventure': ['adventure', 'expedition', 'explore', 'journey', 'quest'],
            'skills': ['survive', 'shelter', 'hunt', 'navigation', 'rescue']
        }
        
        found_skills = []
        overview_lower = overview.lower()
        for skill_type, words in survival_keywords.items():
            if any(word in overview_lower for word in words):
                found_skills.append(skill_type)
        
        if found_skills:
            return f"\n**SURVIVAL & ADVENTURE SKILLS:**\nFound: {', '.join(found_skills)}\nLearn about: Outdoor survival, emergency preparedness, navigation"
        else:
            return f"\n**SURVIVAL SKILLS:**\nNo significant survival content found"
    
    def _analyze_medical_scenes(self, overview, keywords):
        medical_keywords = {
            'medicine': ['doctor', 'hospital', 'medical', 'surgery', 'disease', 'treatment'],
            'psychology': ['psychology', 'mental', 'therapy', 'psychiatrist', 'mind'],
            'emergency': ['emergency', 'paramedic', 'ambulance', 'rescue', 'trauma'],
            'research': ['research', 'experiment', 'cure', 'vaccine', 'outbreak']
        }
        
        found_medical = []
        overview_lower = overview.lower()
        for field, words in medical_keywords.items():
            if any(word in overview_lower for word in words):
                found_medical.append(field)
        
        if found_medical:
            return f"\n**MEDICAL & HEALTHCARE CONTENT:**\nFound: {', '.join(found_medical)}\nLearn about: Medical procedures, healthcare systems, human biology\nNote: Movie medical scenes are often dramatized - verify with real sources"
        else:
            return f"\n**MEDICAL CONTENT:**\nNo significant medical content found"
    
    def _analyze_technology_scenes(self, overview, keywords):
        tech_keywords = {
            'computers': ['computer', 'hack', 'code', 'programming', 'software', 'algorithm'],
            'robotics': ['robot', 'ai', 'artificial intelligence', 'machine', 'android'],
            'cybersecurity': ['hacker', 'cyber', 'security', 'encryption', 'data'],
            'invention': ['invention', 'technology', 'innovate', 'create', 'build']
        }
        
        found_tech = []
        overview_lower = overview.lower()
        for field, words in tech_keywords.items():
            if any(word in overview_lower for word in words):
                found_tech.append(field)
        
        if found_tech:
            return f"\n**TECHNOLOGY & COMPUTER SCIENCE:**\nFound: {', '.join(found_tech)}\nLearn about: Computer programming, AI, cybersecurity, robotics\nReality check: Movie hacking is often exaggerated for drama"
        else:
            return f"\n**TECHNOLOGY CONTENT:**\nNo significant tech content found"
    
    def _calculate_educational_value(self, overview, keywords):
        educational_words = [
            'learn', 'study', 'education', 'knowledge', 'discover', 'research',
            'science', 'history', 'mathematics', 'physics', 'biology', 'chemistry',
            'skill', 'technique', 'method', 'process', 'system', 'theory',
            'doctor', 'scientist', 'teacher', 'engineer', 'researcher', 'expert'
        ]
        
        overview_lower = overview.lower()
        keywords_lower = keywords.lower()
        score = sum(1 for word in educational_words if word in overview_lower or word in keywords_lower)
        max_score = len(educational_words)
        educational_percentage = (score / max_score) * 100 if max_score > 0 else 0
        
        if educational_percentage >= 70:
            rating = "HIGH - Great for learning!"
        elif educational_percentage >= 40:
            rating = "MEDIUM - Some educational content"
        else:
            rating = "LOW - Primarily entertainment"
        
        return f"\n**OVERALL EDUCATIONAL VALUE:** {educational_percentage:.1f}%\nRating: {rating}"
    
    def _find_movie_data(self, movie_name):
        list_of_all_titles = self.movies_data['title'].tolist()
        find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles, n=1, cutoff=0.3)
        
        if not find_close_match:
            return None
        
        close_match = find_close_match[0]
        movie_idx = self.movies_data[self.movies_data['title'] == close_match].index
        
        if len(movie_idx) == 0:
            return None
        
        return self.movies_data.loc[movie_idx[0]]

class CareerInspirationExtractor:
    def __init__(self, movies_data):
        self.movies_data = movies_data
    
    def extract_career_inspiration(self, movie_title):
        movie_data = self._find_movie_data(movie_title)
        if movie_data is None:
            return None, f"Movie '{movie_title}' not found."
        
        overview = str(movie_data.get('overview', ''))
        keywords = str(movie_data.get('keywords', ''))
        cast = str(movie_data.get('cast', ''))
        genres = str(movie_data.get('genres', ''))
        
        careers = self._analyze_career_paths(overview, keywords, cast, genres)
        
        skills = self._analyze_professional_skills(overview, keywords, cast)
        
        inspiration_score = self._calculate_inspiration_score(careers, skills)
        
        report = []
        report.append(f"**CAREER INSPIRATION: {movie_data['title']}**")
        report.append("="*50)
        report.append("\n**CAREER PATHS FOUND:**")
        
        if careers:
            for i, career in enumerate(careers, 1):
                report.append(f"{i}. {career}")
                if career == 'Scientist':
                    report.append("   - Research and discovery focus")
                    report.append("   - Analytical thinking and experimentation")
                elif career == 'Engineer':
                    report.append("   - Problem-solving and design")
                    report.append("   - Technical innovation")
                elif career == 'Leader':
                    report.append("   - Strategic decision making")
                    report.append("   - Team management and direction")
                elif career == 'Doctor':
                    report.append("   - Medical knowledge and patient care")
                    report.append("   - Emergency response skills")
        else:
            report.append("No specific career paths identified")
            report.append("This movie focuses more on general life experiences")
        
        report.append("\n**PROFESSIONAL SKILLS DEMONSTRATED:**")
        if skills:
            for skill in skills:
                report.append(f"• {skill}")
        else:
            report.append("General life skills and personal development")
        
        report.append("\n**EDUCATIONAL PATHS TO EXPLORE:**")
        if any(career in careers for career in ['Scientist', 'Engineer', 'Astronaut', 'Researcher']):
            report.append("• STEM fields (Science, Technology, Engineering, Mathematics)")
            report.append("• University degrees in relevant sciences")
            report.append("• Research internships and laboratory experience")
        
        if any(career in careers for career in ['Doctor', 'Psychologist', 'Researcher']):
            report.append("• Medical or psychology degrees")
            report.append("• Healthcare certifications")
            report.append("• Clinical experience and residencies")
        
        if any(career in careers for career in ['Leader', 'Entrepreneur', 'Diplomat']):
            report.append("• Business administration degrees")
            report.append("• Leadership development programs")
            report.append("• Public speaking and negotiation courses")
        
        if any(career in careers for career in ['Artist', 'Writer', 'Musician']):
            report.append("• Arts and humanities degrees")
            report.append("• Creative workshops and portfolio development")
            report.append("• Internships in creative industries")
        
        report.append(f"\n**CAREER INSPIRATION SCORE: {inspiration_score}/100**")
        
        if inspiration_score >= 70:
            report.append("Rating: HIGHLY INSPIRING - Great for career exploration!")
            report.append("This movie showcases diverse professional paths and skills")
        elif inspiration_score >= 40:
            report.append("Rating: MODERATELY INSPIRING - Good career insights")
            report.append("Contains valuable professional lessons")
        elif inspiration_score >= 20:
            report.append("Rating: MILDLY INSPIRING - Some career relevance")
            report.append("Focuses more on personal than professional development")
        else:
            report.append("Rating: ENTERTAINMENT FOCUSED - Limited career content")
            report.append("Primarily for enjoyment rather than career inspiration")
        
        return "\n".join(report), None
    
    def _analyze_career_paths(self, overview, keywords, cast, genres):
        career_keywords = {
            'Scientist': ['scientist', 'research', 'laboratory', 'experiment', 'discovery', 'physics', 'biology', 'chemistry'],
            'Engineer': ['engineer', 'technology', 'build', 'design', 'invent', 'mechanical', 'electrical'],
            'Astronaut': ['astronaut', 'space', 'nasa', 'rocket', 'mission', 'exploration'],
            'Programmer': ['programmer', 'coder', 'hacker', 'software', 'algorithm', 'computer', 'code'],
            'Doctor': ['doctor', 'surgeon', 'medical', 'hospital', 'treatment', 'diagnose'],
            'Psychologist': ['psychologist', 'therapist', 'mental', 'counselor', 'psychiatry'],
            'Researcher': ['researcher', 'scientist', 'lab', 'study', 'analysis', 'data'],
            'Artist': ['artist', 'painter', 'sculptor', 'creative', 'designer', 'artistic'],
            'Writer': ['writer', 'author', 'journalist', 'reporter', 'novel', 'story'],
            'Musician': ['musician', 'singer', 'band', 'music', 'composer', 'concert'],
            'Leader': ['leader', 'manager', 'director', 'executive', 'ceo', 'president', 'king', 'queen'],
            'Entrepreneur': ['entrepreneur', 'business', 'startup', 'company', 'founder'],
            'Diplomat': ['diplomat', 'ambassador', 'negotiate', 'peace', 'treaty', 'government'],
            'Explorer': ['explorer', 'adventurer', 'expedition', 'discover', 'journey', 'quest'],
            'Archaeologist': ['archaeologist', 'artifact', 'ancient', 'dig', 'history', 'ruins'],
            'Environmentalist': ['environmental', 'conservation', 'nature', 'wildlife', 'planet', 'eco'],
            'Lawyer': ['lawyer', 'attorney', 'court', 'legal', 'judge', 'justice'],
            'Detective': ['detective', 'investigator', 'solve', 'mystery', 'clue', 'case'],
            'Police': ['police', 'officer', 'cop', 'law enforcement', 'agent']
        }
        
        found_careers = []
        combined_text = (overview + ' ' + keywords + ' ' + cast + ' ' + genres).lower()
        
        for career, keywords_list in career_keywords.items():
            if any(keyword in combined_text for keyword in keywords_list):
                found_careers.append(career)
        
        return found_careers
    
    def _analyze_professional_skills(self, overview, keywords, cast):
        skill_categories = {
            'Leadership': ['lead', 'manage', 'direct', 'command', 'strategize', 'decision'],
            'Problem Solving': ['solve', 'fix', 'resolve', 'troubleshoot', 'analyze', 'investigate'],
            'Communication': ['communicate', 'speak', 'negotiate', 'persuade', 'present', 'explain'],
            'Teamwork': ['team', 'collaborate', 'partner', 'work together', 'cooperate'],
            'Creativity': ['create', 'design', 'invent', 'innovate', 'imagine', 'artistic'],
            'Technical': ['technical', 'engineer', 'program', 'build', 'operate', 'technical'],
            'Analytical': ['analyze', 'research', 'study', 'examine', 'evaluate', 'data'],
            'Courage': ['brave', 'courage', 'risk', 'danger', 'fearless', 'bold'],
            'Adaptability': ['adapt', 'flexible', 'change', 'adjust', 'survive', 'improvise']
        }
        
        found_skills = []
        combined_text = (overview + ' ' + keywords).lower()
        
        for skill, keywords_list in skill_categories.items():
            if any(keyword in combined_text for keyword in keywords_list):
                found_skills.append(skill)
        
        return found_skills
    
    def _calculate_inspiration_score(self, careers, skills):
        career_score = len(careers) * 10
        skill_score = len(skills) * 5
        total_score = career_score + skill_score
        return min(total_score, 100)
    
    def _find_movie_data(self, movie_name):
        list_of_all_titles = self.movies_data['title'].tolist()
        find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles, n=1, cutoff=0.3)
        
        if not find_close_match:
            return None
        
        close_match = find_close_match[0]
        movie_idx = self.movies_data[self.movies_data['title'] == close_match].index
        
        if len(movie_idx) == 0:
            return None
        
        return self.movies_data.loc[movie_idx[0]]

@st.cache_resource
def initialize_classes(_df):
    search_engine = SmartMovieSearch(_df)
    skill_builder = MovieSkillBuilder(_df)
    career_extractor = CareerInspirationExtractor(_df)
    return search_engine, skill_builder, career_extractor

search_engine, skill_builder, career_extractor = initialize_classes(df)

st.sidebar.image("download__98_-removebg-preview.png", width=150)
st.sidebar.title("🍿 Choose Your Scene")

feature = st.sidebar.selectbox(
    "Choose Feature",
    [
        "🎞️ Dashboard Overview",
        "🔍 SMART MOVIE SEARCH",
        "🎯 Get Movie Recommendations",
        "📊 Show Dataset Analytics",
        "📈 K-Means Clustering Analysis",
        "🎬🆚🎬 Compare Two Movies",
        "🗺️ Movie Discovery Engine",
        "😊 Mood Based Recommendations",
        "🧠 Movie Skill Builder",
        "💼 Career Inspiration Extractor"
    ]
)

if feature == "🎞️ Dashboard Overview":
    st.header("📊 Dashboard Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Movies", len(df))
    with col2:
        if 'vote_average' in df.columns:
            avg_rating = df['vote_average'].mean()
            st.metric("Avg Rating", f"{avg_rating:.1f}/10")
        else:
            st.metric("Avg Rating", "N/A")
    with col3:
        if 'release_year' in df.columns and not df['release_year'].isna().all():
            years = f"{int(df['release_year'].min())}-{int(df['release_year'].max())}"
            st.metric("Year Range", years)
        else:
            st.metric("Year Range", "N/A")
    with col4:
        if 'mood' in df.columns and not df['mood'].isna().all():
            top_mood = df['mood'].mode()[0]
            st.metric("Top Mood", top_mood)
        else:
            st.metric("Top Mood", "N/A")
    
    st.markdown("---")
    
    st.subheader("🚀 Available Features")
    features = [
        ("🔍 SMART MOVIE SEARCH", "Advanced search with filters, autocomplete, and smart matching"),
        ("🎯 Get Movie Recommendations", "Content-based recommendations using TF-IDF"),
        ("📊 Dataset Analytics", "Comprehensive statistics and visualizations"),
        ("📈 K-Means Clustering", "Group movies into clusters based on features"),
        ("🎬🆚🎬 Compare Two Movies", "Side-by-side comparison with similarity score"),
        ("🗺️ Movie Discovery Engine", "Discover movies by genre, director, decade"),
        ("😊 Mood-Based Recommendations", "Find movies based on emotional mood"),
        ("🧠 Movie Skill Builder", "Extract educational skills from movies"),
        ("💼 Career Inspiration Extractor", "Discover career paths from movies")
    ]
    
    cols = st.columns(3)
    for idx, (title, desc) in enumerate(features):
        with cols[idx % 3]:
            st.markdown(f"""
            <div class='feature-card'>
                <h4>{title}</h4>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.subheader("🎬 Recent Movies Preview")
    if 'release_year' in df.columns:
        recent_movies = df.sort_values('release_year', ascending=False).head(10)
        preview_cols = [c for c in ['title', 'release_year', 'director', 'vote_average', 'genres'] if c in df.columns]
        st.dataframe(
            recent_movies[preview_cols],
            use_container_width=True
        )

elif feature == "🔍 SMART MOVIE SEARCH":
    st.header("🔍 Smart Movie Search Engine")
    st.markdown("Find movies by: Title, Director, Cast, Genres, Keywords with instant results and smart matching")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        search_query = st.text_input("Enter your search query:", placeholder="e.g., Christopher Nolan action movies")
    with col2:
        max_results = st.number_input("Max results", min_value=5, max_value=50, value=20)
    
    if search_query and len(search_query) >= 2:
        suggestions = search_engine.get_autocomplete_suggestions(search_query, max_suggestions=5)
        if suggestions:
            with st.expander("💡 Suggestions"):
                for suggestion in suggestions:
                    st.write(f"• {suggestion}")
    
    with st.expander("⚙️ Advanced Filters", expanded=False):
        filter_col1, filter_col2 = st.columns(2)
        
        with filter_col1:
            if search_query and len(search_query.strip()) > 2:
                matching_movies = []
                query_lower = search_query.lower().strip()
                
                for idx, movie in df.iterrows():
                    title = str(movie.get('title', '')).lower()
                    if query_lower in title or title in query_lower:
                        year = None
                        if 'release_year' in movie and pd.notna(movie['release_year']):
                            try:
                                year = int(float(movie['release_year']))
                                if 1900 <= year <= 2030:
                                    matching_movies.append({
                                        'title': movie.get('title', 'Unknown'),
                                        'year': year
                                    })
                            except:
                                continue
                
                if matching_movies:
                    unique_years = sorted(set([m['year'] for m in matching_movies if m['year']]))
                    if len(unique_years) > 1:
                        year_options = ["Any year"] + [f"{year} (from search results)" for year in unique_years]
                        selected_year = st.selectbox(
                            "Release Year (from search results)", 
                            options=year_options, 
                            index=0,
                            help="Shows years only from movies matching your search query"
                        )
                    else:
                        years = []
                        for year in df['release_year'].dropna():
                            try:
                                year_int = int(float(year))
                                if 1900 <= year_int <= 2030:
                                    years.append(year_int)
                            except:
                                continue
                        
                        if years:
                            unique_years_all = sorted(set(years))
                            year_options = ["Any year"] + [str(y) for y in unique_years_all]
                            selected_year = st.selectbox("Release Year", options=year_options, index=0)
                        else:
                            selected_year = "Any year"
                else:
                    years = []
                    for year in df['release_year'].dropna():
                        try:
                            year_int = int(float(year))
                            if 1900 <= year_int <= 2030:
                                years.append(year_int)
                        except:
                            continue
                    
                    if years:
                        unique_years_all = sorted(set(years))
                        year_options = ["Any year"] + [str(y) for y in unique_years_all]
                        selected_year = st.selectbox("Release Year", options=year_options, index=0)
                    else:
                        selected_year = "Any year"
            else:
                years = []
                for year in df['release_year'].dropna():
                    try:
                        year_int = int(float(year))
                        if 1900 <= year_int <= 2030:
                            years.append(year_int)
                    except:
                        continue
                
                if years:
                    unique_years_all = sorted(set(years))
                    year_options = ["Any year"] + [str(y) for y in unique_years_all]
                    selected_year = st.selectbox("Release Year", options=year_options, index=0)
                else:
                    selected_year = "Any year"
            
            min_rating = st.slider("Minimum rating", 0.0, 10.0, 6.0, 0.1)
        
        with filter_col2:
            max_runtime = st.number_input("Max runtime (minutes)", min_value=60, max_value=300, value=180)
            
            if 'original_language' in df.columns:
                languages = sorted(df['original_language'].dropna().unique().tolist())
                selected_language = st.selectbox("Language", options=["Any language"] + languages)
            else:
                selected_language = "Any language"
            
            sort_by = st.selectbox(
                "Sort results by",
                options=["Relevance", "Rating (High to Low)", "Year (New to Old)", "Year (Old to New)", "Popularity"]
            )
    
    filters = {
        'min_rating': min_rating,
        'max_runtime': max_runtime
    }
    
    if selected_year != "Any year":
        try:
            if "(from search results)" in selected_year:
                year_int = int(selected_year.split()[0])
            else:
                year_int = int(selected_year)
            filters['year'] = year_int
        except:
            pass
    
    if selected_language != "Any language":
        filters['language'] = selected_language
    
    if sort_by != "Relevance":
        filters['sort_by'] = sort_by
    
    if st.button("🔎 Search", type="primary"):
        if search_query:
            with st.spinner("Searching..."):
                results = search_engine.smart_search(search_query, filters, max_results)
                
                if results:
                    st.success(f"Found {len(results)} movies")
                    
                    if 'sort_by' in filters:
                        if filters['sort_by'] == "Rating (High to Low)":
                            results = sorted(results, key=lambda x: x['movie'].get('vote_average', 0), reverse=True)
                        elif filters['sort_by'] == "Year (New to Old)":
                            results = sorted(results, 
                                           key=lambda x: search_engine._get_release_year(x['movie']) or 0, 
                                           reverse=True)
                        elif filters['sort_by'] == "Year (Old to New)":
                            results = sorted(results, 
                                           key=lambda x: search_engine._get_release_year(x['movie']) or 0)
                        elif filters['sort_by'] == "Popularity" and 'popularity' in df.columns:
                            results = sorted(results, 
                                           key=lambda x: x['movie'].get('popularity', 0), 
                                           reverse=True)
                    
                    for i, result in enumerate(results, 1):
                        movie = result['movie']
                        score = result['score']
                        match_type = result['match_type']
                        
                        with st.expander(f"{i}. **{movie['title']}** (Score: {score:.3f})"):
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                year = search_engine._get_release_year(movie)
                                st.write(f"**Year:** {year}")
                                st.write(f"**Director:** {movie.get('director', 'N/A')}")
                                
                                rating = movie.get('vote_average', 'N/A')
                                if isinstance(rating, (int, float)):
                                    st.write(f"**Rating:** {rating:.1f}/10")
                                else:
                                    st.write(f"**Rating:** {rating}")
                                
                                if 'genres' in movie and pd.notna(movie['genres']):
                                    genres = str(movie['genres'])
                                    genres = re.sub(r'[\[\]{}"\']', '', genres)
                                    st.write(f"**Genres:** {genres[:100]}")
                                
                                st.write(f"**Match Type:** {match_type}")
                                
                                if 'overview' in movie and pd.notna(movie['overview']):
                                    overview = str(movie['overview'])
                                    if len(overview) > 200:
                                        overview = overview[:200] + "..."
                                    st.write(f"**Overview:** {overview}")
                            
                            with col2:
                                st.metric("Similarity", f"{score:.3f}")
                                if 'budget' in movie and isinstance(movie.get('budget'), (int, float)) and movie['budget'] > 0:
                                    st.metric("Budget", f"${movie['budget']/1e6:.1f}M")
                                if 'revenue' in movie and isinstance(movie.get('revenue'), (int, float)) and movie['revenue'] > 0:
                                    st.metric("Revenue", f"${movie['revenue']/1e6:.1f}M")
                    
                    if search_query:
                        result_years = []
                        for result in results:
                            year = search_engine._get_release_year(result['movie'])
                            if year and year not in result_years:
                                result_years.append(year)
                        
                        if len(result_years) > 1:
                            st.info(f"📅 **Multiple versions found:** This search returned movies from {len(result_years)} different years: {', '.join(map(str, sorted(result_years)))}. Use the year filter to select a specific version.")
                            
                else:
                    st.warning("No results found. Try a different search query or relax filters.")
        else:
            st.warning("Please enter a search query!")

    if 'search_query' in locals() and search_query:
        common_series_keywords = ['avatar', 'star wars', 'harry potter', 'marvel', 'avengers', 
                                 'spider-man', 'batman', 'superman', 'fast and furious', 
                                 'mission impossible', 'jurassic', 'transformers']
        
        query_lower = search_query.lower()
        is_series = any(keyword in query_lower for keyword in common_series_keywords)
        
        if is_series:
            st.divider()
            st.subheader("🎬 Series/Movie Franchise Helper")
            
            series_movies = []
            for idx, movie in df.iterrows():
                title = str(movie.get('title', '')).lower()
                if any(keyword in title for keyword in common_series_keywords if keyword in query_lower):
                    year = search_engine._get_release_year(movie)
                    if year:
                        series_movies.append({
                            'title': movie.get('title', 'Unknown'),
                            'year': year,
                            'rating': movie.get('vote_average', 0)
                        })
            
            if series_movies:
                series_df = pd.DataFrame(series_movies)
                series_df = series_df.sort_values('year')
                
                st.write(f"**Movies in this series/franchise:**")
                
                years_in_series = series_df['year'].unique()
                selected_series_year = st.selectbox(
                    "Quickly filter to a specific year:",
                    options=["Show all"] + [str(year) for year in years_in_series]
                )
                
                st.dataframe(
                    series_df[['title', 'year', 'rating']].rename(
                        columns={'title': 'Movie Title', 'year': 'Year', 'rating': 'Rating'}
                    ),
                    use_container_width=True
                )
                
                if selected_series_year != "Show all":
                    st.info(f"💡 Tip: Copy '{search_query}' and select year {selected_series_year} in the filter above to search for that specific movie.")

elif feature == "🎯 Get Movie Recommendations":
    st.header("🌟 Movie Recommendations")
    
    @st.cache_resource
    def build_recommendation_system(_df):
        df_copy = _df.copy()
        
        df_copy['combined_features'] = (
            df_copy['genres'].fillna('') + ' ' +
            df_copy['keywords'].fillna('') + ' ' +
            df_copy['director'].fillna('') + ' ' +
            df_copy['cast'].fillna('') + ' ' +
            df_copy['overview'].fillna('')
        )
        
        tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        tfidf_matrix = tfidf.fit_transform(df_copy['combined_features'])
        
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        return tfidf, cosine_sim
    
    tfidf, cosine_sim = build_recommendation_system(df)
    
    movie_list = df['title'].tolist()
    selected_movie = st.selectbox("Select a movie you like:", options=movie_list)
    
    num_recommendations = st.slider("Number of recommendations", 5, 20, 10)
    
    if st.button("Get Recommendations", type="primary"):
        try:
            movie_idx = df[df['title'] == selected_movie].index[0]
            
            similarity_scores = list(enumerate(cosine_sim[movie_idx]))
            
            similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
            
            recommendations = []
            for idx, score in similarity_scores[1:num_recommendations+1]:
                movie = df.iloc[idx]
                recommendations.append({
                    'title': movie['title'],
                    'similarity': float(score),
                    'year': movie.get('release_year', 'N/A'),
                    'rating': movie.get('vote_average', 'N/A'),
                    'director': movie.get('director', 'N/A'),
                    'genres': str(movie.get('genres', 'N/A'))[:50],
                    'overview': movie.get('overview', '')
                })
            
            if recommendations:
                st.success(f"Found {len(recommendations)} similar movies")
                
                for i, rec in enumerate(recommendations, 1):
                    with st.expander(f"{i}. {rec['title']} (Similarity: {rec['similarity']:.3f})"):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(f"**Year:** {rec['year']}")
                            st.write(f"**Director:** {rec['director']}")
                            
                            rating = rec['rating']
                            if isinstance(rating, (int, float)):
                                st.write(f"**Rating:** {rating:.1f}/10")
                            else:
                                st.write(f"**Rating:** {rating}")
                            
                            st.write(f"**Genres:** {rec['genres']}")
                            
                            if rec['overview'] and len(str(rec['overview'])) > 10:
                                overview = str(rec['overview'])
                                if len(overview) > 200:
                                    overview = overview[:200] + "..."
                                st.write(f"**Overview:** {overview}")
                        
                        with col2:
                            similarity_percent = int(rec['similarity'] * 100)
                            st.metric("Similarity", f"{similarity_percent}%")
                            st.progress(similarity_percent / 100)
                            
                            if similarity_percent >= 80:
                                st.success("Strong Match")
                            elif similarity_percent >= 60:
                                st.info("Good Match")
                            else:
                                st.warning("Fair Match")
            else:
                st.warning("No similar movies found. Try selecting a different movie.")
                
        except Exception as e:
            st.error(f"Error: {str(e)}")

elif feature == "📊 Show Dataset Analytics":
    st.header("📊 Dataset Analytics")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Ratings", "🎭 Genres", "📅 Years", "💰 Finance"])
    
    with tab1:
        st.subheader("📈 Detailed Rating Statistics")
        
        if 'vote_average' in df.columns:
            ratings_series = pd.to_numeric(df['vote_average'], errors='coerce').fillna(0)
            
            total_movies = len(df)
            movies_with_ratings = len(ratings_series[ratings_series > 0])
            zero_rating_count = len(ratings_series[ratings_series == 0])
            zero_rating_percent = (zero_rating_count / total_movies) * 100
            avg_rating = ratings_series[ratings_series > 0].mean()
            min_rating = ratings_series[ratings_series > 0].min()
            max_rating = ratings_series.max()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Movies", total_movies)
                st.metric("Movies with Ratings", movies_with_ratings)
            
            with col2:
                st.metric("Average Rating", f"{avg_rating:.2f}/10")
                st.metric("Rating Range", f"{min_rating:.1f} - {max_rating:.1f}")
            
            with col3:
                st.metric("Zero Ratings", f"{zero_rating_count} ({zero_rating_percent:.1f}%)")
                st.metric("Median Rating", f"{ratings_series[ratings_series > 0].median():.2f}/10")
            
            st.info(f"""
            **Summary:**
            - {movies_with_ratings} out of {total_movies} movies have ratings ({movies_with_ratings/total_movies*100:.1f}%)
            - {zero_rating_count} movies ({zero_rating_percent:.1f}%) have a rating of 0
            - The average rating among rated movies is {avg_rating:.2f}/10
            """)
            
            st.subheader("📊 Rating Distribution")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            non_zero_ratings = ratings_series[ratings_series > 0]
            ax1.hist(non_zero_ratings, bins=15, edgecolor='black', alpha=0.7, color='skyblue')
            ax1.set_xlabel('Rating')
            ax1.set_ylabel('Count')
            ax1.set_title('Distribution (Excluding Zeros)')
            ax1.grid(True, alpha=0.3)
            
            ax2.boxplot(non_zero_ratings, vert=False, patch_artist=True)
            ax2.set_xlabel('Rating')
            ax2.set_title('Rating Spread')
            ax2.grid(True, alpha=0.3, axis='x')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.subheader("🏆 Rating Categories")
            
            rating_categories = {
                'Excellent (8.0+)': (8.0, 10.0),
                'Good (6.0-7.9)': (6.0, 7.9),
                'Average (4.0-5.9)': (4.0, 5.9),
                'Poor (<4.0)': (0.1, 3.9)
            }
            
            category_counts = {}
            for category, (low, high) in rating_categories.items():
                count = len(non_zero_ratings[(non_zero_ratings >= low) & (non_zero_ratings <= high)])
                percentage = (count / len(non_zero_ratings)) * 100
                category_counts[category] = (count, percentage)
            
            cat_col1, cat_col2, cat_col3, cat_col4 = st.columns(4)
            categories = list(category_counts.keys())
            
            with cat_col1:
                count, percent = category_counts.get(categories[0], (0, 0))
                st.metric(categories[0], f"{count} ({percent:.1f}%)")
            
            with cat_col2:
                count, percent = category_counts.get(categories[1], (0, 0))
                st.metric(categories[1], f"{count} ({percent:.1f}%)")
            
            with cat_col3:
                count, percent = category_counts.get(categories[2], (0, 0))
                st.metric(categories[2], f"{count} ({percent:.1f}%)")
            
            with cat_col4:
                count, percent = category_counts.get(categories[3], (0, 0))
                st.metric(categories[3], f"{count} ({percent:.1f}%)")
            
        else:
            st.warning("No rating data available in the dataset.")
    
    with tab2:
        if 'genres' in df.columns:
            st.subheader("🎭 Genre Analysis")
            
            all_genres = []
            for genres in df['genres'].dropna():
                if isinstance(genres, str):
                    clean_genres = re.sub(r'[\[\]{}"\']', '', genres)
                    genre_list = [genre.strip() for genre in clean_genres.split(',') if genre.strip()]
                    all_genres.extend(genre_list)
            
            if all_genres:
                genre_counts = pd.Series(all_genres).value_counts()
                total_genres = len(all_genres)
                unique_genres = len(genre_counts)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Genre Tags", total_genres)
                with col2:
                    st.metric("Unique Genres", unique_genres)
                with col3:
                    st.metric("Avg Genres per Movie", f"{total_genres/len(df):.1f}")
                
                top_genres = genre_counts.head(12)
                
                fig, ax = plt.subplots(figsize=(10, 5))
                bars = ax.barh(range(len(top_genres)), top_genres.values)
                ax.set_yticks(range(len(top_genres)))
                ax.set_yticklabels(top_genres.index)
                ax.set_xlabel('Number of Movies')
                ax.set_title('Top 12 Most Common Genres')
                ax.invert_yaxis()
                
                for i, v in enumerate(top_genres.values):
                    ax.text(v + 0.5, i, str(v), va='center')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                st.subheader("📋 Genre Breakdown")
                genre_df = pd.DataFrame({
                    'Genre': genre_counts.index,
                    'Count': genre_counts.values,
                    'Percentage': (genre_counts.values / total_genres * 100).round(1)
                })
                st.dataframe(genre_df.head(15), use_container_width=True)
            else:
                st.info("No genre data extracted.")
        else:
            st.info("No genre data available.")
    
    with tab3:
        if 'release_year' in df.columns:
            st.subheader("📅 Year Distribution Analysis")
            
            years = pd.to_numeric(df['release_year'], errors='coerce').dropna()
            years = years[years.between(1900, 2024)].astype(int)
            
            if not years.empty:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Earliest Year", int(years.min()))
                with col2:
                    st.metric("Latest Year", int(years.max()))
                with col3:
                    st.metric("Year Span", f"{int(years.max() - years.min())} years")
                with col4:
                    st.metric("Total Years Covered", len(years.unique()))
                
                col5, col6, col7, col8 = st.columns(4)
                with col5:
                    st.metric("Average Movies/Year", f"{len(years)/len(years.unique()):.1f}")
                with col6:
                    most_productive_year = years.value_counts().idxmax()
                    st.metric("Most Productive Year", int(most_productive_year))
                with col7:
                    movies_in_peak_year = years.value_counts().max()
                    st.metric("Movies in Peak Year", movies_in_peak_year)
                with col8:
                    median_year = int(years.median())
                    st.metric("Median Year", median_year)
                
                st.subheader("📈 Movies Released by Year")
                
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
                
                year_counts = years.value_counts().sort_index()
                ax1.bar(year_counts.index, year_counts.values, alpha=0.7, color='steelblue', edgecolor='black')
                ax1.set_xlabel('Release Year')
                ax1.set_ylabel('Number of Movies')
                ax1.set_title('Movies Released by Year')
                ax1.grid(True, alpha=0.3)
                
                try:
                    z = np.polyfit(year_counts.index, year_counts.values, 3)
                    p = np.poly1d(z)
                    x_smooth = np.linspace(year_counts.index.min(), year_counts.index.max(), 100)
                    ax1.plot(x_smooth, p(x_smooth), 'r-', linewidth=2, label='Trend')
                    ax1.legend()
                except:
                    pass
                
                moving_avg = year_counts.rolling(window=5, center=True).mean()
                ax2.plot(year_counts.index, year_counts.values, 'o-', alpha=0.5, label='Yearly Count', markersize=3)
                ax2.plot(moving_avg.index, moving_avg.values, 'r-', linewidth=2, label='5-Year Moving Avg')
                ax2.set_xlabel('Release Year')
                ax2.set_ylabel('Number of Movies')
                ax2.set_title('Yearly Production with Moving Average')
                ax2.grid(True, alpha=0.3)
                ax2.legend()
                
                plt.tight_layout()
                st.pyplot(fig)
                
                st.subheader("📅 Movies by Decade")
                
                df_copy = df.copy()
                df_copy['release_year'] = pd.to_numeric(df_copy['release_year'], errors='coerce')
                df_copy = df_copy.dropna(subset=['release_year'])
                df_copy['decade'] = (df_copy['release_year'] // 10) * 10
                decade_counts = df_copy['decade'].value_counts().sort_index()
                
                decade_df = pd.DataFrame({
                    'Decade': decade_counts.index,
                    'Movies': decade_counts.values,
                    'Growth_Rate': decade_counts.pct_change() * 100,
                    'Market_Share': (decade_counts.values / decade_counts.values.sum() * 100).round(1)
                })
                
                dec_col1, dec_col2 = st.columns(2)
                
                with dec_col1:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    bars = ax.bar(range(len(decade_counts)), decade_counts.values, color='teal', alpha=0.7)
                    ax.set_xticks(range(len(decade_counts)))
                    ax.set_xticklabels([f"{int(d)}s" for d in decade_counts.index], rotation=45)
                    ax.set_ylabel('Number of Movies')
                    ax.set_title('Movies by Decade')
                    
                    for i, v in enumerate(decade_counts.values):
                        ax.text(i, v + 0.5, str(v), ha='center')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                
                with dec_col2:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    wedges, texts, autotexts = ax.pie(
                        decade_df['Movies'], 
                        labels=decade_df['Decade'].astype(str) + 's',
                        autopct='%1.1f%%',
                        startangle=90,
                        colors=plt.cm.Set3(np.linspace(0, 1, len(decade_df)))
                    )
                    ax.set_title('Decade Market Share (%)')
                    plt.tight_layout()
                    st.pyplot(fig)
                
                st.subheader("📊 Decade Statistics")
                
                display_df = decade_df.copy()
                
                display_df['Decade'] = display_df['Decade'].astype(int).astype(str) + 's'
                
                display_df['Growth_Rate'] = display_df['Growth_Rate'].apply(
                    lambda x: f"{x:+.1f}%" if not pd.isna(x) else "—"
                )
                
                display_df['Market_Share'] = display_df['Market_Share'].apply(
                    lambda x: f"{x:.1f}%"
                )
                
                display_df.columns = ['Decade', 'Movies', 'Growth Rate', 'Market Share']
                
                def color_growth_rate(val):
                    if isinstance(val, str) and val != "—":
                        try:
                            num_val = float(val.replace('%', '').replace('+', ''))
                            if num_val > 5:
                                return 'background-color: #d4edda; color: #155724; font-weight: bold'
                            elif num_val > 0:
                                return 'background-color: #d1ecf1; color: #0c5460'
                            elif num_val < -5:
                                return 'background-color: #f8d7da; color: #721c24; font-weight: bold'
                            elif num_val < 0:
                                return 'background-color: #fff3cd; color: #856404'
                        except:
                            pass
                    return ''
                
                def highlight_top_market(val):
                    if isinstance(val, str):
                        try:
                            share_val = float(val.replace('%', ''))
                            if share_val == display_df['Market Share'].apply(
                                lambda x: float(x.replace('%', '')) if isinstance(x, str) else 0
                            ).max():
                                return 'background-color: #e7f3ff; border-left: 4px solid #007bff; font-weight: bold'
                        except:
                            pass
                    return ''
                
                styled_df = (
                    display_df.style
                    .applymap(color_growth_rate, subset=['Growth Rate'])
                    .applymap(highlight_top_market, subset=['Market Share'])
                    .set_properties(**{
                        'text-align': 'center',
                        'border': '1px solid #dee2e6'
                    })
                    .set_table_styles([
                        {'selector': 'th', 'props': [
                            ('background-color', '#f8f9fa'),
                            ('color', '#212529'),
                            ('font-weight', 'bold'),
                            ('text-align', 'center'),
                            ('border-bottom', '2px solid #dee2e6'),
                            ('padding', '8px')
                        ]},
                        {'selector': 'td', 'props': [
                            ('padding', '8px'),
                            ('border-top', '1px solid #dee2e6')
                        ]},
                        {'selector': 'tr:hover', 'props': [
                            ('background-color', '#f5f5f5')
                        ]}
                    ])
                )
                
                st.dataframe(styled_df, use_container_width=True)
                
                st.caption("""
                🎨 **Color Key:**
                • 📈 **Dark Green**: Strong Growth (>5%)  
                • 📗 **Light Green**: Positive Growth  
                • 📉 **Dark Red**: Significant Decline (>5%)  
                • 📙 **Light Orange**: Mild Decline  
                • 🔷 **Blue Highlight**: Highest Market Share
                """)
                
                st.subheader("📈 Yearly Trends Analysis")
                
                yearly_stats = pd.DataFrame({'year': years})
                
                available_metrics = {}
                if 'vote_average' in df.columns:
                    df['year_temp'] = pd.to_numeric(df['release_year'], errors='coerce')
                    yearly_ratings = df.groupby('year_temp')['vote_average'].mean().dropna()
                    available_metrics['Average Rating'] = yearly_ratings
                
                if 'runtime' in df.columns:
                    df['year_temp'] = pd.to_numeric(df['release_year'], errors='coerce')
                    yearly_runtime = df.groupby('year_temp')['runtime'].mean().dropna()
                    available_metrics['Average Runtime (min)'] = yearly_runtime
                
                if available_metrics:
                    n_metrics = len(available_metrics)
                    fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 4 * n_metrics))
                    
                    if n_metrics == 1:
                        axes = [axes]
                    
                    for idx, (metric_name, metric_data) in enumerate(available_metrics.items()):
                        axes[idx].plot(metric_data.index, metric_data.values, 'o-', markersize=3, linewidth=1)
                        axes[idx].set_xlabel('Year')
                        axes[idx].set_ylabel(metric_name)
                        axes[idx].set_title(f'{metric_name} by Year')
                        axes[idx].grid(True, alpha=0.3)
                        
                        try:
                            z = np.polyfit(metric_data.index, metric_data.values, 1)
                            p = np.poly1d(z)
                            axes[idx].plot(metric_data.index, p(metric_data.index), 'r--', 
                                          linewidth=1, alpha=0.7, label='Trend')
                            axes[idx].legend()
                        except:
                            pass
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                
                st.subheader("🏆 Peak Production Years")
                
                top_years = years.value_counts().head(10)
                
                fig, ax = plt.subplots(figsize=(10, 4))
                bars = ax.barh(range(len(top_years)), top_years.values, color='orange', alpha=0.7)
                ax.set_yticks(range(len(top_years)))
                ax.set_yticklabels([f"{int(year)}" for year in top_years.index])
                ax.set_xlabel('Number of Movies')
                ax.set_title('Top 10 Most Productive Years')
                ax.invert_yaxis()
                
                for i, v in enumerate(top_years.values):
                    ax.text(v + 0.5, i, str(v), va='center')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                st.subheader("🔍 Production Clusters by Era")
                
                era_bins = [1900, 1950, 1970, 1990, 2000, 2010, 2025]
                era_labels = ['Pre-1950', '1950-69', '1970-89', '1990-99', '2000-09', '2010+']
                
                years_series = pd.Series(years)
                era_counts = pd.cut(years_series, bins=era_bins, labels=era_labels).value_counts().sort_index()
                
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.bar(range(len(era_counts)), era_counts.values, color='purple', alpha=0.7)
                ax.set_xticks(range(len(era_counts)))
                ax.set_xticklabels(era_counts.index, rotation=45)
                ax.set_ylabel('Number of Movies')
                ax.set_title('Movies by Historical Era')
                
                for i, v in enumerate(era_counts.values):
                    ax.text(i, v + 0.5, str(v), ha='center')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                st.subheader("📈 Simple Production Forecast")
                
                if len(years) > 20:
                    recent_years = years[years >= 2000]
                    if len(recent_years) > 10:
                        year_counts_recent = recent_years.value_counts().sort_index()
                        
                        fig, ax = plt.subplots(figsize=(10, 4))
                        ax.plot(year_counts_recent.index, year_counts_recent.values, 'o-', label='Actual')
                        
                        try:
                            x = np.array(year_counts_recent.index)
                            y = np.array(year_counts_recent.values)
                            z = np.polyfit(x, y, 1)
                            p = np.poly1d(z)
                            
                            future_years = np.arange(x[-1] + 1, x[-1] + 4)
                            ax.plot(future_years, p(future_years), 'r--', label='Projection')
                            ax.legend()
                            ax.set_title('Production Trend with 3-Year Projection')
                        except:
                            ax.set_title('Production Trend')
                        
                        ax.set_xlabel('Year')
                        ax.set_ylabel('Movies Produced')
                        ax.grid(True, alpha=0.3)
                        plt.tight_layout()
                        st.pyplot(fig)
                
                st.subheader("📋 Year Distribution Summary")
                
                percentiles = years.quantile([0.25, 0.5, 0.75, 0.9]).astype(int)
                
                summary_text = f"""
                **Year Distribution Summary:**
                
                **Basic Statistics:**
                - Dataset covers {len(years.unique())} years ({int(years.min())} - {int(years.max())})
                - Median release year: {int(years.median())}
                - Average of {len(years)/len(years.unique()):.1f} movies per year
                
                **Distribution Insights:**
                - 25% of movies were released before {percentiles[0.25]}
                - 50% of movies were released before {percentiles[0.5]}
                - 75% of movies were released before {percentiles[0.75]}
                - 90% of movies were released before {percentiles[0.9]}
                
                **Peak Production:**
                - Peak year: {most_productive_year} ({movies_in_peak_year} movies)
                - Top decade: {int(decade_counts.idxmax())}s ({decade_counts.max()} movies)
                
                **Growth Patterns:**
                - Production peaked in the {int(most_productive_year//10)*10}s decade
                - Current trend: {'Increasing' if years.mean() > years.median() else 'Stable/Decreasing'}
                """
                
                st.info(summary_text)
                
            else:
                st.info("No valid year data for analysis.")
        else:
            st.info("No release year data available.")
    
    with tab4:
        if 'budget' in df.columns and 'revenue' in df.columns:
            st.subheader("💰 Financial Analysis")
            
            financial_df = df.copy()
            financial_df['budget'] = pd.to_numeric(financial_df['budget'], errors='coerce')
            financial_df['revenue'] = pd.to_numeric(financial_df['revenue'], errors='coerce')
            financial_df = financial_df[(financial_df['budget'] > 0) & (financial_df['revenue'] > 0)]
            
            if len(financial_df) > 10:
                total_movies_financial = len(financial_df)
                avg_budget = financial_df['budget'].mean() / 1e6
                avg_revenue = financial_df['revenue'].mean() / 1e6
                avg_profit = (financial_df['revenue'] - financial_df['budget']).mean() / 1e6
                avg_roi = ((financial_df['revenue'] - financial_df['budget']) / financial_df['budget'] * 100).mean()
                
                profitable_movies = len(financial_df[financial_df['revenue'] > financial_df['budget']])
                profit_percentage = (profitable_movies / len(financial_df)) * 100
                median_profit = (financial_df['revenue'] - financial_df['budget']).median() / 1e6
                breakeven_ratio = len(financial_df[financial_df['revenue'] >= financial_df['budget']]) / len(financial_df) * 100
                
                st.write("**📊 Key Financial Metrics**")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Movies with Financial Data", total_movies_financial)
                    st.metric("Avg Budget", f"${avg_budget:.1f}M")
                
                with col2:
                    st.metric("Avg Revenue", f"${avg_revenue:.1f}M")
                    st.metric("Avg Profit", f"${avg_profit:.1f}M")
                
                with col3:
                    st.metric("Profitable Movies", f"{profitable_movies} ({profit_percentage:.1f}%)")
                    st.metric("Average ROI", f"{avg_roi:.1f}%")
                
                with col4:
                    st.metric("Median Profit", f"${median_profit:.1f}M")
                    st.metric("Break-even Rate", f"{breakeven_ratio:.1f}%")
                
                st.subheader("📈 Budget vs Revenue Analysis")
                
                financial_df['profit'] = financial_df['revenue'] - financial_df['budget']
                financial_df['roi'] = ((financial_df['profit']) / financial_df['budget'] * 100)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                
                budget_millions = financial_df['budget'] / 1e6
                revenue_millions = financial_df['revenue'] / 1e6
                
                scatter = ax.scatter(
                    budget_millions,
                    revenue_millions,
                    c=financial_df['roi'],
                    cmap='RdYlGn',
                    alpha=0.7,
                    s=50,
                    edgecolors='black',
                    linewidth=0.5
                )
                
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('ROI %', rotation=270, labelpad=20)
                
                max_val = max(budget_millions.max(), revenue_millions.max())
                ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, alpha=0.7, label='Break-even')
                
                ax.set_xlabel('Budget ($ Millions)', fontweight='bold')
                ax.set_ylabel('')
                ax.set_title('Budget vs Revenue Analysis\n(Color: ROI %)', fontweight='bold', fontsize=14)
                
                ax.set_xticks([0, 50, 100, 150, 200, 250])
                ax.set_xlim(0, 250)
                
                ax.set_yticks([0, 500, 1000, 1500, 2000, 2500, 3000])
                ax.set_ylim(0, 3000)
                
                ax.grid(True, alpha=0.3, linestyle='-')
                
                budget_revenue_corr = financial_df['budget'].corr(financial_df['revenue'])
                
                if budget_revenue_corr > 0.7:
                    corr_strength = "Very Strong"
                elif budget_revenue_corr > 0.5:
                    corr_strength = "Strong"
                elif budget_revenue_corr > 0.3:
                    corr_strength = "Moderate"
                else:
                    corr_strength = "Weak"
                
                stats_text = f"""Correlation: {budget_revenue_corr:.3f}
Strong positive correlation
Above line = Profitable
Below line = Unprofitable
Sample: {len(financial_df)} movies"""
                
                ax.text(0.02, 0.98, stats_text,
                       transform=ax.transAxes,
                       bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
                       verticalalignment='top',
                       fontsize=10,
                       fontweight='bold')
                
                ax.legend()
                plt.tight_layout()
                st.pyplot(fig)
                
                st.write(f"**📊 Financial Insights (based on {len(financial_df)} movies):**")
                
                insights_col1, insights_col2 = st.columns(2)
                
                with insights_col1:
                    st.info(f"""
                    **Correlation Analysis:**
                    • Budget-Revenue Correlation: {budget_revenue_corr:.3f} ({corr_strength})
                    • Average Budget: ${avg_budget:.1f}M
                    • Average Revenue: ${avg_revenue:.1f}M
                    • Average Profit: ${avg_profit:.1f}M
                    """)
                
                with insights_col2:
                    profitable_count = len(financial_df[financial_df['profit'] > 0])
                    break_even_count = len(financial_df[financial_df['profit'] == 0])
                    loss_count = len(financial_df[financial_df['profit'] < 0])
                    
                    st.info(f"""
                    **Profitability Breakdown:**
                    • Profitable Movies: {profitable_count} ({profitable_count/len(financial_df)*100:.1f}%)
                    • Break-even Movies: {break_even_count} ({break_even_count/len(financial_df)*100:.1f}%)
                    • Loss-making Movies: {loss_count} ({loss_count/len(financial_df)*100:.1f}%)
                    """)
                
                if profitable_count > 0:
                    top_profitable = financial_df.nlargest(3, 'profit')
                    st.write("**🏆 Top 3 Most Profitable Movies:**")
                    
                    top_display = []
                    for i, (idx, movie) in enumerate(top_profitable.iterrows(), 1):
                        title = movie.get('title', movie.get('original_title', f"Movie {idx}"))
                        budget_m = movie['budget'] / 1e6
                        revenue_m = movie['revenue'] / 1e6
                        profit_m = movie['profit'] / 1e6
                        roi_pct = ((revenue_m - budget_m) / budget_m * 100) if budget_m > 0 else 0
                        
                        top_display.append({
                            'Rank': i,
                            'Movie': title,
                            'Budget ($M)': f"{budget_m:.1f}",
                            'Revenue ($M)': f"{revenue_m:.1f}",
                            'Profit ($M)': f"{profit_m:.1f}",
                            'ROI (%)': f"{roi_pct:.1f}"
                        })
                    
                    if top_display:
                        top_df = pd.DataFrame(top_display)
                        st.dataframe(top_df.set_index('Rank'), use_container_width=True)
                
                st.subheader("💰 Budget & Revenue Tiers")
                
                budget_tiers = {
                    'Micro (<$1M)': (0, 1_000_000),
                    'Low ($1-10M)': (1_000_000, 10_000_000),
                    'Medium ($10-50M)': (10_000_000, 50_000_000),
                    'High ($50-100M)': (50_000_000, 100_000_000),
                    'Blockbuster (>$100M)': (100_000_000, float('inf'))
                }
                
                tier_data = []
                for tier_name, (min_budget, max_budget) in budget_tiers.items():
                    tier_movies = financial_df[(financial_df['budget'] >= min_budget) & 
                                              (financial_df['budget'] < max_budget)]
                    if len(tier_movies) > 0:
                        avg_roi_tier = ((tier_movies['revenue'] - tier_movies['budget']) / tier_movies['budget'] * 100).mean()
                        success_rate = (len(tier_movies[tier_movies['revenue'] > tier_movies['budget']]) / len(tier_movies)) * 100
                        avg_profit_tier = (tier_movies['revenue'] - tier_movies['budget']).mean() / 1e6
                        tier_data.append({
                            'Budget Tier': tier_name,
                            'Movies': len(tier_movies),
                            'Avg Budget ($M)': f"{tier_movies['budget'].mean() / 1e6:.1f}",
                            'Avg ROI': f"{avg_roi_tier:.1f}%",
                            'Success Rate': f"{success_rate:.1f}%",
                            'Avg Profit ($M)': f"{avg_profit_tier:.1f}"
                        })
                
                if tier_data:
                    tier_df = pd.DataFrame(tier_data)
                    
                    def color_tier_performance(val):
                        if isinstance(val, str) and '%' in val:
                            num_val = float(val.replace('%', ''))
                            if 'ROI' in val:
                                if num_val > 50:
                                    return 'background-color: #d4edda; color: #155724; font-weight: bold'
                                elif num_val > 0:
                                    return 'background-color: #d1ecf1; color: #0c5460'
                                else:
                                    return 'background-color: #f8d7da; color: #721c24'
                            elif 'Success' in val:
                                if num_val > 70:
                                    return 'background-color: #d4edda; color: #155724; font-weight: bold'
                                elif num_val > 50:
                                    return 'background-color: #d1ecf1; color: #0c5460'
                                else:
                                    return 'background-color: #f8d7da; color: #721c24'
                        return ''
                    
                    styled_tier_df = tier_df.style.applymap(color_tier_performance, 
                                                          subset=['Avg ROI', 'Success Rate'])
                    st.dataframe(styled_tier_df, use_container_width=True)
                
                st.subheader("💡 Financial Summary")
                
                total_investment = financial_df['budget'].sum() / 1e9
                total_revenue = financial_df['revenue'].sum() / 1e9
                total_profit = (financial_df['revenue'].sum() - financial_df['budget'].sum()) / 1e9
                
                summary_col1, summary_col2 = st.columns(2)
                
                with summary_col1:
                    st.success(f"""
                    **Overall Performance:**
                    • Total Investment: ${total_investment:.2f}B
                    • Total Revenue: ${total_revenue:.2f}B
                    • Total Profit: ${total_profit:.2f}B
                    • Overall ROI: {(total_profit/total_investment*100) if total_investment > 0 else 0:.1f}%
                    """)
                
                with summary_col2:
                    st.warning(f"""
                    **Key Insights:**
                    • {profit_percentage:.1f}% of movies are profitable
                    • Average ROI: {avg_roi:.1f}%
                    • Budget and revenue show {corr_strength.lower()} positive correlation
                    • Higher budgets tend to have higher revenue potential
                    """)
                
            else:
                st.info(f"Not enough financial data for comprehensive analysis. Only {len(financial_df)} movies have valid budget and revenue data.")
                
                if len(financial_df) > 0:
                    st.write("**Available Financial Data Preview:**")
                    preview_df = financial_df[['budget', 'revenue']].head()
                    preview_df['budget'] = preview_df['budget'] / 1e6
                    preview_df['revenue'] = preview_df['revenue'] / 1e6
                    preview_df.columns = ['Budget ($M)', 'Revenue ($M)']
                    st.dataframe(preview_df, use_container_width=True)
        else:
            st.info("Budget and/or revenue columns not available in this dataset.")

elif feature == "📈 K-Means Clustering Analysis":
    st.header("📈 K-Means Clustering Analysis")
    st.info("This feature groups movies into clusters based on selected numeric features.")
    
    available_features = []
    for col in ['vote_average', 'vote_count', 'popularity', 'budget', 'revenue', 'runtime']:
        if col in df.columns:
            available_features.append(col)
    
    selected_features = st.multiselect(
        "Select features for clustering:",
        options=available_features,
        default=[f for f in ['vote_average', 'popularity'] if f in available_features]
        if len(available_features) >= 2 else available_features
    )
    
    if len(selected_features) >= 2:
        n_clusters = st.slider("Number of clusters:", 2, 10, 4)
        
        if st.button("Run Clustering", type="primary"):
            with st.spinner("Clustering in progress..."):
                try:
                    cluster_data = df[selected_features].copy()
                    cluster_data = cluster_data.apply(
                        lambda x: pd.to_numeric(x, errors='coerce')
                    )
                    cluster_data = cluster_data.fillna(cluster_data.mean())
                    
                    scaler = StandardScaler()
                    scaled_data = scaler.fit_transform(cluster_data)
                    
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    clusters = kmeans.fit_predict(scaled_data)
                    
                    df_clustered = df.copy()
                    df_clustered['cluster'] = clusters
                    
                    st.success(f"Created {n_clusters} clusters")
                    
                    st.subheader("Cluster Sizes")
                    cluster_sizes = pd.Series(clusters).value_counts().sort_index()
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    cluster_sizes.plot(kind='bar', ax=ax)
                    ax.set_xlabel('Cluster')
                    ax.set_ylabel('Number of Movies')
                    ax.set_title('Movies per Cluster')
                    for i, v in enumerate(cluster_sizes):
                        ax.text(i, v + 0.5, str(v), ha='center')
                    st.pyplot(fig)
                    
                    st.subheader("Cluster Characteristics")
                    for cluster_id in range(n_clusters):
                        with st.expander(
                            f"Cluster {cluster_id} ({cluster_sizes.get(cluster_id, 0)} movies)"
                        ):
                            cluster_movies = df_clustered[df_clustered['cluster'] == cluster_id]
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                if 'vote_average' in selected_features:
                                    st.write(
                                        f"**Avg Rating:** "
                                        f"{cluster_movies['vote_average'].mean():.2f}"
                                    )
                                if 'popularity' in selected_features:
                                    st.write(
                                        f"**Avg Popularity:** "
                                        f"{cluster_movies['popularity'].mean():.2f}"
                                    )
                            with col2:
                                if 'budget' in selected_features:
                                    st.write(
                                        f"**Avg Budget:** "
                                        f"${cluster_movies['budget'].mean()/1e6:.1f}M"
                                    )
                                if 'revenue' in selected_features:
                                    st.write(
                                        f"**Avg Revenue:** "
                                        f"${cluster_movies['revenue'].mean()/1e6:.1f}M"
                                    )
                            
                            sample_cols = [c for c in ['title', 'release_year', 'director'] if c in cluster_movies.columns]
                            if not cluster_movies.empty and sample_cols:
                                st.write("**Sample Movies:**")
                                st.dataframe(
                                    cluster_movies[sample_cols].head(5),
                                    use_container_width=True
                                )
                    
                    st.subheader("Cluster Visualization")
                    
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=2)
                    pca_result = pca.fit_transform(scaled_data)
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    scatter = ax.scatter(
                        pca_result[:, 0],
                        pca_result[:, 1],
                        c=clusters,
                        alpha=0.6,
                        s=40
                    )
                    ax.set_xlabel('PCA Component 1')
                    ax.set_ylabel('PCA Component 2')
                    ax.set_title('Movie Clusters (2D PCA Projection)')
                    plt.colorbar(scatter, ax=ax, label='Cluster')
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"Clustering error: {str(e)}")
    else:
        st.warning("Please select at least 2 numeric features for clustering.")

elif feature == "🎬🆚🎬 Compare Two Movies":
    st.header("🎬🆚🎬 Compare Two Movies")
    
    def calculate_robust_similarity(movie1, movie2):
        similarity_score = 0.0
        weight_sum = 0.0
        
        if 'genres' in movie1 and 'genres' in movie2:
            genres1 = str(movie1['genres']).lower()
            genres2 = str(movie2['genres']).lower()
            
            if genres1 and genres2 and genres1 != 'nan' and genres2 != 'nan':
                import re
                genres1_list = re.findall(r'[a-z]+', genres1)
                genres2_list = re.findall(r'[a-z]+', genres2)
                
                if genres1_list and genres2_list:
                    common_genres = len(set(genres1_list) & set(genres2_list))
                    total_genres = len(set(genres1_list) | set(genres2_list))
                    genre_similarity = common_genres / total_genres if total_genres > 0 else 0
                    similarity_score += genre_similarity * 0.3
                    weight_sum += 0.3
        
        if 'vote_average' in movie1 and 'vote_average' in movie2:
            rating1 = movie1.get('vote_average', 0)
            rating2 = movie2.get('vote_average', 0)
            
            try:
                rating1 = float(rating1) if pd.notna(rating1) else 0
                rating2 = float(rating2) if pd.notna(rating2) else 0
                
                if rating1 > 0 and rating2 > 0:
                    rating_diff = abs(rating1 - rating2) / 10.0
                    rating_similarity = 1.0 - rating_diff
                    similarity_score += rating_similarity * 0.2
                    weight_sum += 0.2
            except:
                pass
        
        if 'release_year' in movie1 and 'release_year' in movie2:
            year1 = movie1.get('release_year', 0)
            year2 = movie2.get('release_year', 0)
            
            try:
                year1 = int(year1) if pd.notna(year1) else 0
                year2 = int(year2) if pd.notna(year2) else 0
                
                if year1 > 1900 and year2 > 1900:
                    year_diff = abs(year1 - year2)
                    year_similarity = 1.0 - (year_diff / 100.0)
                    year_similarity = max(0, min(1, year_similarity))
                    similarity_score += year_similarity * 0.15
                    weight_sum += 0.15
            except:
                pass
        
        if 'runtime' in movie1 and 'runtime' in movie2:
            runtime1 = movie1.get('runtime', 0)
            runtime2 = movie2.get('runtime', 0)
            
            try:
                runtime1 = float(runtime1) if pd.notna(runtime1) else 0
                runtime2 = float(runtime2) if pd.notna(runtime2) else 0
                
                if runtime1 > 0 and runtime2 > 0:
                    runtime_diff = abs(runtime1 - runtime2) / 180.0
                    runtime_similarity = 1.0 - runtime_diff
                    runtime_similarity = max(0, min(1, runtime_similarity))
                    similarity_score += runtime_similarity * 0.1
                    weight_sum += 0.1
            except:
                pass
        
        if 'popularity' in movie1 and 'popularity' in movie2:
            pop1 = movie1.get('popularity', 0)
            pop2 = movie2.get('popularity', 0)
            
            try:
                pop1 = float(pop1) if pd.notna(pop1) else 0
                pop2 = float(pop2) if pd.notna(pop2) else 0
                
                if pop1 > 0 and pop2 > 0:
                    max_pop = max(pop1, pop2)
                    if max_pop > 0:
                        pop1_scaled = pop1 / max_pop
                        pop2_scaled = pop2 / max_pop
                        pop_similarity = 1.0 - abs(pop1_scaled - pop2_scaled)
                        similarity_score += pop_similarity * 0.1
                        weight_sum += 0.1
            except:
                pass
        
        if 'budget' in movie1 and 'budget' in movie2:
            budget1 = movie1.get('budget', 0)
            budget2 = movie2.get('budget', 0)
            
            try:
                budget1 = float(budget1) if pd.notna(budget1) else 0
                budget2 = float(budget2) if pd.notna(budget2) else 0
                
                if budget1 > 0 and budget2 > 0:
                    import math
                    budget1_log = math.log10(budget1) if budget1 > 0 else 0
                    budget2_log = math.log10(budget2) if budget2 > 0 else 0
                    
                    if budget1_log > 0 and budget2_log > 0:
                        budget_diff = abs(budget1_log - budget2_log) / 3.0
                        budget_similarity = 1.0 - budget_diff
                        budget_similarity = max(0, min(1, budget_similarity))
                        similarity_score += budget_similarity * 0.05
                        weight_sum += 0.05
            except:
                pass
        
        title1 = str(movie1.get('title', '')).lower()
        title2 = str(movie2.get('title', '')).lower()
        
        if 'original_title' in movie1:
            title1_words = set(re.findall(r'[a-z]+', title1 + ' ' + str(movie1.get('original_title', '')).lower()))
        else:
            title1_words = set(re.findall(r'[a-z]+', title1))
            
        if 'original_title' in movie2:
            title2_words = set(re.findall(r'[a-z]+', title2 + ' ' + str(movie2.get('original_title', '')).lower()))
        else:
            title2_words = set(re.findall(r'[a-z]+', title2))
        
        if title1_words and title2_words:
            common_words = len(title1_words & title2_words)
            total_words = len(title1_words | title2_words)
            title_similarity = common_words / total_words if total_words > 0 else 0
            similarity_score += title_similarity * 0.1
            weight_sum += 0.1
        
        if weight_sum > 0:
            final_score = similarity_score / weight_sum
        else:
            final_score = 0.0
        
        return final_score
    
    movie_list = df['title'].tolist()
    
    col1, col2 = st.columns(2)
    with col1:
        movie1 = st.selectbox("Select first movie:", options=movie_list, key="movie1")
    with col2:
        movie2_options = [m for m in movie_list if m != movie1]
        movie2 = st.selectbox("Select second movie:", options=movie2_options, key="movie2")
    
    if st.button("Compare Movies", type="primary"):
        try:
            movie1_data = df[df['title'] == movie1].iloc[0].to_dict()
            movie2_data = df[df['title'] == movie2].iloc[0].to_dict()
            
            similarity_score = calculate_robust_similarity(movie1_data, movie2_data)
            
            st.subheader("📊 Comparison Results")
            
            col_score, col_analysis = st.columns([1, 2])
            
            with col_score:
                st.metric("Similarity Score", f"{similarity_score:.3f}")
                
                if similarity_score >= 0.8:
                    st.success("**🎯 Very Similar**")
                    st.caption("Likely same genre, director, or theme")
                elif similarity_score >= 0.6:
                    st.info("**👍 Similar**")
                    st.caption("Shared many characteristics")
                elif similarity_score >= 0.4:
                    st.warning("**🤔 Somewhat Similar**")
                    st.caption("Some common elements")
                elif similarity_score >= 0.2:
                    st.error("**👎 Not Very Similar**")
                    st.caption("Few common elements")
                else:
                    st.error("**❌ Very Different**")
                    st.caption("Different genres and themes")
            
            with col_analysis:
                st.write("**📈 Similarity Analysis:**")
                
                similarities = []
                
                if 'genres' in movie1_data and 'genres' in movie2_data:
                    genres1 = str(movie1_data['genres'])
                    genres2 = str(movie2_data['genres'])
                    if genres1 != 'nan' and genres2 != 'nan':
                        import re
                        genres1_list = re.findall(r'[a-z]+', genres1.lower())
                        genres2_list = re.findall(r'[a-z]+', genres2.lower())
                        common_genres = set(genres1_list) & set(genres2_list)
                        if common_genres:
                            similarities.append(f"🎭 **Shared genres:** {', '.join(common_genres)}")
                
                if 'release_year' in movie1_data and 'release_year' in movie2_data:
                    year1 = movie1_data.get('release_year')
                    year2 = movie2_data.get('release_year')
                    if pd.notna(year1) and pd.notna(year2):
                        year_diff = abs(int(year1) - int(year2))
                        if year_diff <= 5:
                            similarities.append(f"📅 **Same era:** Released within {year_diff} years")
                
                if 'vote_average' in movie1_data and 'vote_average' in movie2_data:
                    rating1 = movie1_data.get('vote_average')
                    rating2 = movie2_data.get('vote_average')
                    if pd.notna(rating1) and pd.notna(rating2):
                        rating_diff = abs(float(rating1) - float(rating2))
                        if rating_diff <= 1.0:
                            similarities.append(f"⭐ **Similar ratings:** {rating1:.1f} vs {rating2:.1f}")
                
                if similarities:
                    for similarity in similarities:
                        st.write(f"• {similarity}")
                else:
                    st.write("• No significant similarities found")
            
            st.subheader("📋 Detailed Comparison")
            
            def fmt_num(x):
                try:
                    return f"{int(x):,}"
                except Exception:
                    return "N/A"
            
            comparison_data = []
            
            comparison_data.append({
                'Category': 'Basic Info',
                'Metric': 'Title',
                movie1[:30]: movie1_data.get('title', 'N/A'),
                movie2[:30]: movie2_data.get('title', 'N/A')
            })
            
            comparison_data.append({
                'Category': 'Basic Info',
                'Metric': 'Release Year',
                movie1[:30]: movie1_data.get('release_year', 'N/A'),
                movie2[:30]: movie2_data.get('release_year', 'N/A')
            })
            
            if 'director' in df.columns:
                comparison_data.append({
                    'Category': 'Basic Info',
                    'Metric': 'Director',
                    movie1[:30]: movie1_data.get('director', 'N/A'),
                    movie2[:30]: movie2_data.get('director', 'N/A')
                })
            
            if 'genres' in df.columns:
                genres1 = str(movie1_data.get('genres', 'N/A'))
                genres2 = str(movie2_data.get('genres', 'N/A'))
                import re
                genres1_clean = re.sub(r'[\[\]{}"\']', '', genres1)
                genres2_clean = re.sub(r'[\[\]{}"\']', '', genres2)
                
                comparison_data.append({
                    'Category': 'Content',
                    'Metric': 'Genres',
                    movie1[:30]: genres1_clean[:50] + ('...' if len(genres1_clean) > 50 else ''),
                    movie2[:30]: genres2_clean[:50] + ('...' if len(genres2_clean) > 50 else '')
                })
            
            comparison_data.append({
                'Category': 'Ratings',
                'Metric': 'Rating',
                movie1[:30]: f"{movie1_data.get('vote_average', 'N/A')}/10",
                movie2[:30]: f"{movie2_data.get('vote_average', 'N/A')}/10"
            })
            
            comparison_data.append({
                'Category': 'Ratings',
                'Metric': 'Vote Count',
                movie1[:30]: fmt_num(movie1_data.get('vote_count', 'N/A')),
                movie2[:30]: fmt_num(movie2_data.get('vote_count', 'N/A'))
            })
            
            if 'runtime' in df.columns:
                comparison_data.append({
                    'Category': 'Technical',
                    'Metric': 'Runtime',
                    movie1[:30]: f"{movie1_data.get('runtime', 'N/A')} min",
                    movie2[:30]: f"{movie2_data.get('runtime', 'N/A')} min"
                })
            
            if 'budget' in df.columns:
                budget1 = movie1_data.get('budget', 0)
                budget2 = movie2_data.get('budget', 0)
                
                comparison_data.append({
                    'Category': 'Financial',
                    'Metric': 'Budget',
                    movie1[:30]: f"${budget1/1e6:.1f}M" if budget1 and float(budget1) > 0 else 'N/A',
                    movie2[:30]: f"${budget2/1e6:.1f}M" if budget2 and float(budget2) > 0 else 'N/A'
                })
            
            if 'revenue' in df.columns:
                revenue1 = movie1_data.get('revenue', 0)
                revenue2 = movie2_data.get('revenue', 0)
                
                comparison_data.append({
                    'Category': 'Financial',
                    'Metric': 'Revenue',
                    movie1[:30]: f"${revenue1/1e6:.1f}M" if revenue1 and float(revenue1) > 0 else 'N/A',
                    movie2[:30]: f"${revenue2/1e6:.1f}M" if revenue2 and float(revenue2) > 0 else 'N/A'
                })
            
            if 'popularity' in df.columns:
                comparison_data.append({
                    'Category': 'Popularity',
                    'Metric': 'Popularity Score',
                    movie1[:30]: f"{movie1_data.get('popularity', 'N/A'):.2f}",
                    movie2[:30]: f"{movie2_data.get('popularity', 'N/A'):.2f}"
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            
            def highlight_similarities(row):
                val1 = str(row[movie1[:30]])
                val2 = str(row[movie2[:30]])
                
                try:
                    num1 = float(''.join(filter(str.isdigit, val1)) or 0)
                    num2 = float(''.join(filter(str.isdigit, val2)) or 0)
                    if num1 > 0 and num2 > 0:
                        diff = abs(num1 - num2) / max(num1, num2)
                        if diff < 0.1:
                            return ['background-color: #d4edda'] * len(row)
                except:
                    pass
                
                if val1 == val2 and val1 != 'N/A':
                    return ['background-color: #d4edda'] * len(row)
                
                return [''] * len(row)
            
            styled_df = comparison_df.style.apply(highlight_similarities, axis=1)
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
            
            st.subheader("📈 Visual Comparison")
            
            metrics = []
            movie1_values = []
            movie2_values = []
            
            try:
                rating1 = float(movie1_data.get('vote_average', 0) or 0)
                rating2 = float(movie2_data.get('vote_average', 0) or 0)
                if rating1 > 0 and rating2 > 0:
                    metrics.append('Rating (/10)')
                    movie1_values.append(rating1)
                    movie2_values.append(rating2)
            except:
                pass
            
            try:
                votes1 = float(movie1_data.get('vote_count', 0) or 0)
                votes2 = float(movie2_data.get('vote_count', 0) or 0)
                if votes1 > 0 and votes2 > 0:
                    metrics.append('Vote Count (scaled)')
                    scale_factor = max(votes1, votes2) / 10
                    movie1_values.append(votes1 / scale_factor)
                    movie2_values.append(votes2 / scale_factor)
            except:
                pass
            
            if 'popularity' in df.columns:
                try:
                    pop1 = float(movie1_data.get('popularity', 0) or 0)
                    pop2 = float(movie2_data.get('popularity', 0) or 0)
                    if pop1 > 0 and pop2 > 0:
                        metrics.append('Popularity (scaled)')
                        scale_factor = max(pop1, pop2) / 10
                        movie1_values.append(pop1 / scale_factor)
                        movie2_values.append(pop2 / scale_factor)
                except:
                    pass
            
            if 'runtime' in df.columns:
                try:
                    runtime1 = float(movie1_data.get('runtime', 0) or 0)
                    runtime2 = float(movie2_data.get('runtime', 0) or 0)
                    if runtime1 > 0 and runtime2 > 0:
                        metrics.append('Runtime (min/30)')
                        movie1_values.append(runtime1 / 30)
                        movie2_values.append(runtime2 / 30)
                except:
                    pass
            
            if metrics:
                x = np.arange(len(metrics))
                width = 0.35
                
                fig, ax = plt.subplots(figsize=(10, 6))
                bars1 = ax.bar(x - width/2, movie1_values, width, label=movie1[:20], color='skyblue')
                bars2 = ax.bar(x + width/2, movie2_values, width, label=movie2[:20], color='lightcoral')
                
                ax.set_xlabel('Metrics')
                ax.set_ylabel('Values (scaled)')
                ax.set_title(f'Comparison: {movie1[:20]} vs {movie2[:20]}')
                ax.set_xticks(x)
                ax.set_xticklabels(metrics)
                ax.legend()
                ax.grid(True, alpha=0.3, axis='y')
                
                for bars in [bars1, bars2]:
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(
                            bar.get_x() + bar.get_width()/2.,
                            height + 0.05,
                            f'{height:.1f}',
                            ha='center',
                            va='bottom',
                            fontsize=9
                        )
                
                plt.tight_layout()
                st.pyplot(fig)
                
                st.subheader("🎯 Recommendation")
                
                if similarity_score >= 0.7:
                    st.success(f"""
                    **🎬 Strong Match!** 
                    If you enjoyed **"{movie1}"**, you'll probably love **"{movie2}"** too!
                    These movies share similar genres, ratings, and overall vibe.
                    """)
                elif similarity_score >= 0.5:
                    st.info(f"""
                    **👍 Good Match**
                    If you liked **"{movie1}"**, **"{movie2}"** could be worth watching.
                    They share some common elements that might appeal to you.
                    """)
                elif similarity_score >= 0.3:
                    st.warning(f"""
                    **🤔 Moderate Match**
                    **"{movie1}"** and **"{movie2}"** have some differences but might still appeal 
                    to the same audience in certain aspects.
                    """)
                else:
                    st.error(f"""
                    **⚠️ Different Experiences**
                    **"{movie1}"** and **"{movie2}"** are quite different. 
                    If you're looking for something similar to **"{movie1}"**, 
                    you might want to try another movie with higher similarity score.
                    """)
                
            else:
                st.info("Not enough comparable metrics available for visualization.")
            
        except Exception as e:
            st.error(f"Comparison error: {str(e)}")
            st.info("Please make sure both movies exist in the dataset and try again.")

elif feature == "🗺️ Movie Discovery Engine":
    st.header("🗺️ Movie Discovery Engine")
    
    discovery_method = st.radio(
        "Choose discovery method:",
        ["Recent Hits", "Genre Deep Dive", "Director Spotlight", "Time Machine (By decade)"]
    )
    
    if discovery_method == "Recent Hits":
        st.subheader("🎬 Recent Hits")
        
        if 'release_year' in df.columns and 'vote_average' in df.columns:
            current_year = datetime.now().year
            recent_years = st.slider("Last N years:", 1, 20, 5)
            
            recent_movies = df[
                (df['release_year'] >= current_year - recent_years) &
                (df['vote_average'] >= 7.0)
            ].copy()
            
            if not recent_movies.empty:
                recent_movies = recent_movies.sort_values(
                    ['vote_average', 'popularity'] if 'popularity' in df.columns else ['vote_average'],
                    ascending=[False, False] if 'popularity' in df.columns else [False]
                )
                st.write(
                    f"Found {len(recent_movies)} recent hits "
                    f"(last {recent_years} years, rating ≥ 7.0):"
                )
                cols = [c for c in ['title', 'release_year', 'vote_average', 'director', 'genres'] if c in recent_movies.columns]
                st.dataframe(recent_movies[cols].head(30), use_container_width=True)
            else:
                st.warning("No recent hits found. Try adjusting the years or rating threshold.")
        else:
            st.info("Required columns 'release_year' and 'vote_average' are missing.")
    
    elif discovery_method == "Genre Deep Dive":
        st.subheader("🎭 Genre Deep Dive")
        
        if 'genres' in df.columns:
            all_genres = []
            for genres in df['genres'].dropna():
                if isinstance(genres, str):
                    clean_genres = re.sub(r'[\[\]{}"\']', '', genres)
                    genre_list = [g.strip() for g in clean_genres.split(',') if g.strip()]
                    all_genres.extend(genre_list)
            unique_genres = sorted(set(all_genres))
            
            selected_genre = st.selectbox("Select a genre:", options=unique_genres)
            
            if selected_genre:
                genre_movies = df[df['genres'].str.contains(selected_genre, na=False)].copy()
                
                sort_by = st.selectbox("Sort by:", ["Rating", "Popularity", "Release Year", "Title"])
                
                if sort_by == "Rating" and 'vote_average' in genre_movies.columns:
                    genre_movies = genre_movies.sort_values('vote_average', ascending=False)
                elif sort_by == "Popularity" and 'popularity' in genre_movies.columns:
                    genre_movies = genre_movies.sort_values('popularity', ascending=False)
                elif sort_by == "Release Year" and 'release_year' in genre_movies.columns:
                    genre_movies = genre_movies.sort_values('release_year', ascending=False)
                else:
                    genre_movies = genre_movies.sort_values('title')
                
                st.write(f"Found {len(genre_movies)} movies in {selected_genre}:")
                cols = [c for c in ['title', 'release_year', 'vote_average', 'director'] if c in genre_movies.columns]
                st.dataframe(genre_movies[cols].head(30), use_container_width=True)
        else:
            st.info("No 'genres' column in dataset.")
    
    elif discovery_method == "Director Spotlight":
        st.subheader("🎥 Director Spotlight")
        
        if 'director' in df.columns:
            director_counts = df['director'].value_counts().head(50)
            selected_director = st.selectbox("Select a director:", options=director_counts.index.tolist())
            
            if selected_director:
                director_movies = df[df['director'] == selected_director].copy()
                if 'release_year' in director_movies.columns:
                    director_movies = director_movies.sort_values('release_year', ascending=False)
                
                st.write(f"Found {len(director_movies)} movies by {selected_director}:")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    if 'vote_average' in director_movies.columns:
                        st.metric("Avg Rating", f"{director_movies['vote_average'].mean():.1f}/10")
                with col2:
                    if 'revenue' in director_movies.columns:
                        st.metric("Total Revenue", f"${director_movies['revenue'].sum()/1e9:.2f}B")
                with col3:
                    if 'release_year' in director_movies.columns and not director_movies['release_year'].isna().all():
                        years = f"{int(director_movies['release_year'].min())}-{int(director_movies['release_year'].max())}"
                        st.metric("Career Span", years)
                
                cols = [c for c in ['title', 'release_year', 'vote_average', 'budget', 'revenue', 'genres'] if c in director_movies.columns]
                st.dataframe(director_movies[cols], use_container_width=True)
        else:
            st.info("No 'director' column in dataset.")
    
    elif discovery_method == "Time Machine (By decade)":
        st.subheader("⏳ Time Machine - Explore by Decade")
        
        if 'release_year' in df.columns:
            decades = {
                '2020s': (2020, 2029),
                '2010s': (2010, 2019),
                '2000s': (2000, 2009),
                '1990s': (1990, 1999),
                '1980s': (1980, 1989),
                '1970s': (1970, 1979),
                '1960s': (1960, 1969),
                '1950s': (1950, 1959),
                '1940s': (1940, 1949),
                '1920s-1930s': (1920, 1939)
            }
            
            selected_decade = st.selectbox("Select a decade:", options=list(decades.keys()))
            
            if selected_decade:
                start_year, end_year = decades[selected_decade]
                decade_movies = df[
                    (df['release_year'] >= start_year) &
                    (df['release_year'] <= end_year)
                ].copy()
                
                if not decade_movies.empty and 'vote_average' in decade_movies.columns:
                    decade_movies = decade_movies.sort_values('vote_average', ascending=False)
                
                st.write(f"Found {len(decade_movies)} movies from {selected_decade}:")
                
                top_movies = decade_movies.head(30)
                for i, (_, movie) in enumerate(top_movies.iterrows(), 1):
                    with st.expander(
                        f"{i}. {movie.get('title', 'Unknown')} "
                        f"({int(movie.get('release_year', 0) or 0)}) - "
                        f"{movie.get('vote_average', 'N/A')}"
                    ):
                        st.write(f"**Director:** {movie.get('director', 'N/A')}")
                        st.write(f"**Genres:** {movie.get('genres', 'N/A')}")
                        if 'overview' in movie and pd.notna(movie['overview']):
                            st.write(f"**Overview:** {str(movie['overview'])[:300]}...")
        else:
            st.info("No 'release_year' column in dataset.")

elif feature == "😊 Mood Based Recommendations":
    st.header("😊 Mood Based Recommendations")
    
    if 'mood' in df.columns:
        moods = df['mood'].dropna().unique().tolist()
        selected_mood = st.selectbox("What mood are you in?", options=moods)
        
        num_movies = st.slider("Number of movies to show:", 5, 30, 15)
        
        if selected_mood:
            mood_movies = df[df['mood'] == selected_mood].copy()
            
            if not mood_movies.empty:
                sort_by = st.radio("Sort by:", ["Rating", "Popularity", "Release Year"])
                
                if sort_by == "Rating" and 'vote_average' in mood_movies.columns:
                    mood_movies = mood_movies.sort_values('vote_average', ascending=False)
                elif sort_by == "Popularity" and 'popularity' in mood_movies.columns:
                    mood_movies = mood_movies.sort_values('popularity', ascending=False)
                elif sort_by == "Release Year" and 'release_year' in mood_movies.columns:
                    mood_movies = mood_movies.sort_values('release_year', ascending=False)
                
                st.success(f"Found {len(mood_movies)} '{selected_mood}' movies")
                
                for i, (_, movie) in enumerate(mood_movies.head(num_movies).iterrows(), 1):
                    def format_year(year_val):
                        if pd.isna(year_val):
                            return "N/A"
                        try:
                            return str(int(year_val))
                        except:
                            return str(year_val)
                    
                    def format_rating(rating_val):
                        if pd.isna(rating_val):
                            return "N/A"
                        try:
                            return f"{float(rating_val):.1f}"
                        except:
                            return str(rating_val)
                    
                    with st.expander(
                        f"{i}. {movie.get('title', 'Unknown')} "
                        f"({format_year(movie.get('release_year'))})"
                    ):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(f"**Director:** {movie.get('director', 'N/A')}")
                            st.write(f"**Rating:** {format_rating(movie.get('vote_average'))}/10")
                            st.write(f"**Genres:** {str(movie.get('genres', 'N/A'))[:120]}...")
                            if 'overview' in movie and pd.notna(movie['overview']):
                                st.write(f"**Overview:** {str(movie['overview'])[:250]}...")
                        with col2:
                            st.metric("Mood", movie.get('mood', 'N/A'))
                            if 'budget' in movie and pd.notna(movie.get('budget')) and movie['budget'] > 0:
                                st.metric("Budget", f"${movie['budget']/1e6:.1f}M")
                
                st.subheader("📊 Mood Statistics")
                mood_counts = df['mood'].value_counts()
                
                fig, ax = plt.subplots(figsize=(10, 6))
                mood_counts.plot(kind='pie', ax=ax, autopct='%1.1f%%', startangle=90)
                ax.set_ylabel('')
                ax.set_title('Distribution of Movie Moods')
                st.pyplot(fig)
            else:
                st.warning(f"No '{selected_mood}' movies found in the dataset.")
    else:
        st.warning("Mood data not available. Please check if your dataset has a 'mood' column.")

elif feature == "🧠 Movie Skill Builder":
    st.header("🧠 Movie Skill Builder")
    st.markdown("Discover educational content and real skills from movies.")
    
    movie_list = df['title'].tolist()
    selected_movie = st.selectbox("Select a movie:", options=movie_list)
    
    if st.button("Analyze Skills", type="primary"):
        with st.spinner("Analyzing movie for educational content..."):
            result, error = skill_builder.extract_skills_from_movie(selected_movie)
            
            if error:
                st.error(error)
            else:
                st.markdown(result)
                
                st.subheader("📊 Skills Overview")
                
                skill_categories = [
                    "COOKING & FOOD SKILLS",
                    "SCIENCE & TECHNOLOGY CONCEPTS",
                    "HISTORICAL CONTEXT",
                    "SURVIVAL & ADVENTURE SKILLS",
                    "MEDICAL & HEALTHCARE CONTENT",
                    "TECHNOLOGY & COMPUTER SCIENCE"
                ]
                
                skill_presence = []
                for category in skill_categories:
                    if category in result:
                        block = result.split(category)[1]
                        line_after = block.split("\n")[1] if "\n" in block else ""
                        if "No significant" not in line_after:
                            skill_presence.append((category, 1))
                        else:
                            skill_presence.append((category, 0))
                
                if skill_presence:
                    categories, presence = zip(*skill_presence)
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    bars = ax.barh(
                        categories,
                        presence
                    )
                    ax.set_xlabel('Presence (1 = Yes, 0 = No)')
                    ax.set_title(f'Skill Categories in "{selected_movie}"')
                    
                    for bar, p in zip(bars, presence):
                        ax.text(
                            bar.get_width() + 0.02,
                            bar.get_y() + bar.get_height()/2,
                            'Present' if p == 1 else 'Not Present',
                            va='center'
                        )
                    
                    st.pyplot(fig)

elif feature == "💼 Career Inspiration Extractor":
    st.header("💼 Career Inspiration Extractor")
    st.markdown("Discover career paths and professional skills from movies.")
    
    movie_list = df['title'].tolist()
    selected_movie = st.selectbox("Select a movie:", options=movie_list, key="career_movie")
    
    if st.button("Extract Career Inspiration", type="primary"):
        with st.spinner("Analyzing movie for career inspiration..."):
            result, error = career_extractor.extract_career_inspiration(selected_movie)
            
            if error:
                st.error(error)
            else:
                st.markdown(result)
                
                lines = result.split('\n')
                careers = []
                skills = []
                inspiration_score = 0
                
                in_career_section = False
                in_skills_section = False
                
                for line in lines:
                    line_stripped = line.strip()
                    
                    if "CAREER INSPIRATION SCORE:" in line_stripped:
                        try:
                            score_text = line_stripped.split("CAREER INSPIRATION SCORE:")[1].strip()
                            score_str = score_text.split("/")[0].strip()
                            inspiration_score = int(score_str)
                        except Exception:
                            pass
                    
                    elif "🎯 CAREER PATHS FOUND:" in line_stripped:
                        in_career_section = True
                        in_skills_section = False
                        continue
                    elif "🛠️ PROFESSIONAL SKILLS DEMONSTRATED:" in line_stripped:
                        in_career_section = False
                        in_skills_section = True
                        continue
                    elif line_stripped.startswith("**Rating:"):
                        in_career_section = False
                        in_skills_section = False
                    
                    if in_career_section and line_stripped:
                        if line_stripped[:2].isdigit() and line_stripped[1] == '.':
                            try:
                                career = line_stripped.split('. ', 1)[1]
                                careers.append(career)
                            except Exception:
                                pass
                        elif "No specific career paths" not in line_stripped:
                            career_keywords = [
                                'Scientist', 'Engineer', 'Astronaut', 'Programmer', 'Doctor',
                                'Psychologist', 'Researcher', 'Artist', 'Writer', 'Musician',
                                'Leader', 'Entrepreneur', 'Diplomat', 'Explorer', 'Archaeologist',
                                'Environmentalist', 'Lawyer', 'Detective', 'Police'
                            ]
                            for keyword in career_keywords:
                                if keyword in line_stripped:
                                    careers.append(keyword)
                                    break
                    
                    elif in_skills_section and line_stripped:
                        if line_stripped.startswith("• "):
                            skills.append(line_stripped[2:])
                        elif "General life skills" not in line_stripped and line_stripped:
                            skill_keywords = [
                                'Leadership', 'Problem Solving', 'Communication', 'Teamwork',
                                'Creativity', 'Technical', 'Analytical', 'Courage', 'Adaptability'
                            ]
                            for keyword in skill_keywords:
                                if keyword in line_stripped:
                                    skills.append(keyword)
                                    break
                
                if not careers:
                    combined_text = ' '.join(lines).lower()
                    career_keywords = {
                        'Scientist': ['scientist', 'research', 'laboratory', 'experiment'],
                        'Engineer': ['engineer', 'technology', 'build', 'design'],
                        'Doctor': ['doctor', 'surgeon', 'medical', 'hospital'],
                        'Artist': ['artist', 'painter', 'creative', 'designer'],
                        'Writer': ['writer', 'author', 'journalist', 'reporter'],
                        'Leader': ['leader', 'manager', 'director', 'executive'],
                        'Detective': ['detective', 'investigator', 'solve', 'mystery']
                    }
                    
                    for career, keywords in career_keywords.items():
                        if any(keyword in combined_text for keyword in keywords):
                            careers.append(career)
                
                if not skills:
                    combined_text = ' '.join(lines).lower()
                    skill_keywords = {
                        'Leadership': ['lead', 'manage', 'direct', 'command'],
                        'Problem Solving': ['solve', 'fix', 'resolve', 'troubleshoot'],
                        'Communication': ['communicate', 'speak', 'negotiate', 'persuade'],
                        'Teamwork': ['team', 'collaborate', 'partner', 'cooperate'],
                        'Creativity': ['create', 'design', 'invent', 'innovate']
                    }
                    
                    for skill, keywords in skill_keywords.items():
                        if any(keyword in combined_text for keyword in keywords):
                            skills.append(skill)
                
                st.subheader("📈 Career Inspiration Visualization")
                
                if careers or skills:
                    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
                    
                    if careers:
                        career_counts = {}
                        for career in careers:
                            career_counts[career] = career_counts.get(career, 0) + 1
                        
                        career_names = list(career_counts.keys())
                        career_values = list(career_counts.values())
                        
                        axes[0].barh(range(len(career_names)), career_values, color='skyblue')
                        axes[0].set_yticks(range(len(career_names)))
                        axes[0].set_yticklabels(career_names)
                        axes[0].set_xlabel('Count')
                        axes[0].set_title('Career Paths Found')
                        
                        for i, v in enumerate(career_values):
                            axes[0].text(v + 0.1, i, str(v), va='center')
                    else:
                        axes[0].text(
                            0.5, 0.5,
                            'No specific\ncareer paths\nidentified',
                            ha='center', va='center',
                            fontsize=12,
                            transform=axes[0].transAxes
                        )
                        axes[0].set_xlim(0, 1)
                        axes[0].set_ylim(0, 1)
                        axes[0].set_title('Career Paths Found')
                    
                    if skills:
                        skill_counts = {}
                        for skill in skills:
                            skill_counts[skill] = skill_counts.get(skill, 0) + 1
                        
                        skill_names = list(skill_counts.keys())
                        skill_values = list(skill_counts.values())
                        
                        axes[1].barh(range(len(skill_names)), skill_values, color='lightgreen')
                        axes[1].set_yticks(range(len(skill_names)))
                        axes[1].set_yticklabels(skill_names)
                        axes[1].set_xlabel('Count')
                        axes[1].set_title('Professional Skills')
                        
                        for i, v in enumerate(skill_values):
                            axes[1].text(v + 0.1, i, str(v), va='center')
                    else:
                        axes[1].text(
                            0.5, 0.5,
                            'General life\nskills and\npersonal development',
                            ha='center', va='center',
                            fontsize=12,
                            transform=axes[1].transAxes
                        )
                        axes[1].set_xlim(0, 1)
                        axes[1].set_ylim(0, 1)
                        axes[1].set_title('Professional Skills')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.info("No specific career paths or skills identified for visualization.")
                
                if inspiration_score > 0:
                    st.subheader("🎯 Inspiration Score")
                    fig, ax = plt.subplots(figsize=(12, 3))
                    
                    gradient = np.linspace(0, 1, 100).reshape(1, -1)
                    ax.imshow(gradient, aspect='auto', cmap='RdYlGn', 
                             extent=[0, 100, 0, 1], alpha=0.7)
                    
                    ax.axvline(x=inspiration_score, linewidth=4, color='black', linestyle='-')
                    
                    ax.text(
                        inspiration_score,
                        0.5,
                        f'Score: {inspiration_score}/100',
                        ha='center',
                        va='center',
                        fontsize=14,
                        fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
                    )
                    
                    categories = {
                        'Entertainment': (0, 20),
                        'Mildly Inspiring': (20, 40),
                        'Moderately Inspiring': (40, 70),
                        'Highly Inspiring': (70, 100)
                    }
                    
                    for category, (start, end) in categories.items():
                        ax.text(
                            (start + end) / 2,
                            -0.2,
                            category,
                            ha='center',
                            va='top',
                            fontsize=10
                        )
                        ax.axvspan(start, end, alpha=0.1, color='gray')
                    
                    ax.set_xlim(0, 100)
                    ax.set_ylim(-0.3, 1.2)
                    ax.set_xlabel('Inspiration Score')
                    ax.set_title(f'Career Inspiration Score for "{selected_movie}"')
                    ax.set_yticks([])
                    ax.grid(True, alpha=0.3, axis='x')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    if inspiration_score >= 70:
                        st.success("**Rating: HIGHLY INSPIRING** - Great for career exploration!")
                    elif inspiration_score >= 40:
                        st.info("**Rating: MODERATELY INSPIRING** - Good career insights")
                    elif inspiration_score >= 20:
                        st.warning("**Rating: MILDLY INSPIRING** - Some career relevance")
                    else:
                        st.error("**Rating: ENTERTAINMENT FOCUSED** - Limited career content")

st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><strong>🎬 CineAI: Smart Movie Analytics & Recommendations</strong></p>
</div>
""", unsafe_allow_html=True)