import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException, StaleElementReferenceException
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
import pandas as pd
import time
import re
import os
import json
from typing import List, Dict
import glob
import stat

class TMDBSuccessfulScraper:
    def __init__(self, api_key: str, headless: bool = False):
        self.api_key = api_key
        self.base_url = "https://www.themoviedb.org"
        self.api_base_url = "https://api.themoviedb.org/3"
        self.setup_driver(headless)
    
    def _is_binary_executable(self, path: str) -> bool:
        """Return True if path is an executable binary (executable bit or ELF header)."""
        try:
            if not os.path.isfile(path):
                return False
            if os.access(path, os.X_OK):
                return True
            with open(path, 'rb') as fh:
                header = fh.read(4)
                return header == b'\x7fELF'
        except Exception:
            return False

    def _resolve_chromedriver_executable(self, installed_path: str) -> str:
        """
        Ensure installed_path points to an executable chromedriver binary.
        If not, search the same directory (and subdirs) for the real executable,
        set chmod +x and return it.
        """
        try:
            # If the path returned is already an executable binary, use it
            if self._is_binary_executable(installed_path):
                return installed_path
        except Exception:
            pass

        driver_dir = os.path.dirname(installed_path) or installed_path

        # First pass: candidates in same dir
        candidates = []
        try:
            for p in glob.glob(os.path.join(driver_dir, '*')):
                name = os.path.basename(p).lower()
                if 'chromedriver' in name:
                    candidates.append(p)
        except Exception:
            candidates = []

        # Check candidates for executable/ELF
        for cand in sorted(candidates):
            if self._is_binary_executable(cand):
                try:
                    os.chmod(cand, 0o755)
                except Exception:
                    pass
                return cand

        # Second pass: walk subdirectories looking for chromedriver file
        for root, _, files in os.walk(driver_dir):
            for f in files:
                if 'chromedriver' in f.lower():
                    p = os.path.join(root, f)
                    if self._is_binary_executable(p):
                        try:
                            os.chmod(p, 0o755)
                        except Exception:
                            pass
                        return p

        # Nothing found - return original so the error is visible
        return installed_path

    def setup_driver(self, headless: bool = False):
        """Setup Chrome driver with options"""
        chrome_options = Options()
        if headless:
            # use modern headless flag
            chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        chrome_options.add_argument("--disable-gpu")
        
        # Install and resolve the correct chromedriver executable
        installed = ChromeDriverManager().install()
        resolved = self._resolve_chromedriver_executable(installed)
        print(f"ChromeDriverManager.install() returned: {installed}")
        print(f"Using chromedriver executable: {resolved}")

        service = Service(resolved)
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        # mask webdriver
        try:
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        except Exception:
            pass
        self.wait = WebDriverWait(self.driver, 15)
    
    def get_all_movies_via_pagination(self, category: str, max_pages: int = 20) -> List[str]:
        """Get ALL movies using working pagination approach"""
        movie_links = set()
        
        print(f"Getting ALL movies from {category} via pagination")
        
        url_patterns = {
            "popular": f"{self.base_url}/movie?page={{}}",
            "now-playing": f"{self.base_url}/movie/now-playing?page={{}}", 
            "upcoming": f"{self.base_url}/movie/upcoming?page={{}}",
            "top-rated": f"{self.base_url}/movie/top-rated?page={{}}"
        }
        
        url_pattern = url_patterns.get(category, f"{self.base_url}/movie?page={{}}")
        
        for page in range(1, max_pages + 1):
            url = url_pattern.format(page)
            print(f"Loading page {page}: {url}")
            
            try:
                self.driver.get(url)
                time.sleep(3)
                
                current_url = self.driver.current_url
                if "404" in current_url or "error" in current_url.lower():
                    print(f"Page {page} not found - stopping")
                    break
                
                links = self._collect_movie_links()
                if not links:
                    print(f"No movies found on page {page} - stopping")
                    break
                
                previous_count = len(movie_links)
                movie_links.update(links)
                new_movies = len(movie_links) - previous_count
                
                if new_movies == 0:
                    print(f"No new movies on page {page} - reached end of content")
                    break
                
                print(f"Page {page}: +{new_movies} new movies, total: {len(movie_links)}")
                
                if new_movies < 5 and page > 3:
                    print(f"Few new movies ({new_movies}) - likely reached end")
                    break
                    
            except Exception as e:
                print(f"Error loading page {page}: {e}")
                break
        
        print(f"Collected {len(movie_links)} total movies from {category}")
        return list(movie_links)
    
    def _collect_movie_links(self) -> List[str]:
        """Collect all valid movie links from current page"""
        movie_links = set()
        
        try:
            movie_elements = self.driver.find_elements(By.CSS_SELECTOR, "a[href*='/movie/']")
            
            for element in movie_elements:
                try:
                    href = element.get_attribute('href')
                    if self._is_valid_movie_link(href):
                        movie_links.add(href)
                except StaleElementReferenceException:
                    continue
                
        except Exception as e:
            print(f"Error collecting links: {e}")
        
        return list(movie_links)
    
    def _is_valid_movie_link(self, href: str) -> bool:
        """Check if href is a valid movie link"""
        if not href:
            return False
        
        if any(x in href for x in ['/cast', '/crew', '/trailers', '/images', '/videos', '/reviews']):
            return False
        
        movie_id_match = re.search(r'/movie/(\d+)-', href)
        if not movie_id_match:
            return False
        
        return True
    
    def get_movie_id_from_url(self, movie_url: str) -> str:
        """Extract movie ID from URL"""
        movie_id_match = re.search(r'/movie/(\d+)', movie_url)
        return movie_id_match.group(1) if movie_id_match else None
    
    def get_complete_movie_data_from_api(self, movie_id: str) -> Dict:
        """Get COMPLETE movie data from TMDB API"""
        try:
            url = f"{self.api_base_url}/movie/{movie_id}"
            params = {
                "api_key": self.api_key,
                "append_to_response": "credits,keywords,release_dates"
            }
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "accept": "application/json"
            }
            
            response = requests.get(url, headers=headers, params=params)
            
            if response.status_code == 200:
                movie_data = response.json()
                
                complete_data = {
                    'index': movie_data.get('id'),
                    'budget': movie_data.get('budget'),
                    'genres': [genre["name"] for genre in movie_data.get("genres", [])],
                    'homepage': movie_data.get('homepage'),
                    'id': movie_data.get('id'),
                    'keywords': [keyword["name"] for keyword in movie_data.get("keywords", {}).get("keywords", [])],
                    'original_language': movie_data.get('original_language'),
                    'original_title': movie_data.get('original_title'),
                    'overview': movie_data.get('overview'),
                    'popularity': movie_data.get('popularity'),
                    'production_companies': [company["name"] for company in movie_data.get("production_companies", [])],
                    'production_countries': [country["name"] for country in movie_data.get("production_countries", [])],
                    'release_date': movie_data.get('release_date'),
                    'revenue': movie_data.get('revenue'),
                    'runtime': movie_data.get('runtime'),
                    'spoken_languages': [lang["english_name"] for lang in movie_data.get("spoken_languages", [])],
                    'status': movie_data.get('status'),
                    'tagline': movie_data.get('tagline'),
                    'title': movie_data.get('title'),
                    'vote_average': movie_data.get('vote_average'),
                    'vote_count': movie_data.get('vote_count')
                }
                
                credits = movie_data.get('credits', {})
                cast = credits.get('cast', [])
                complete_data['cast'] = [f"{actor['name']} as {actor.get('character', 'N/A')}" for actor in cast[:15]]
                
                crew = credits.get('crew', [])
                directors = [person for person in crew if person.get("job") == "Director"]
                complete_data['director'] = [director["name"] for director in directors]
                
                important_crew = []
                for person in crew:
                    if person.get("job") in ["Director", "Producer", "Screenplay", "Writer"]:
                        important_crew.append(f"{person['name']} ({person['job']})")
                complete_data['crew'] = important_crew[:10]
                
                return complete_data
                
            else:
                print(f"API error for movie {movie_id}: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"API extraction error for movie {movie_id}: {e}")
            return None
    
    def scrape_all_categories_complete(self, pages_per_category: int = 20) -> pd.DataFrame:
        """Scrape ALL movies from all categories using pagination"""
        categories = {
            "popular": "Popular Movies",
            "now-playing": "Now Playing", 
            "upcoming": "Upcoming",
            "top-rated": "Top Rated"
        }
        
        all_movies_data = []
        
        print("Starting SUCCESSFUL movie scraping via PAGINATION")
        print(f"Target: {pages_per_category} pages per category")
        print("Browser and API initialized successfully")
        
        for category_key, category_name in categories.items():
            print(f"\n{'='*60}")
            print(f"PAGINATION - {category_name.upper()}")
            print(f"{'='*60}")
            
            movie_links = self.get_all_movies_via_pagination(category_key, pages_per_category)
            print(f"Found {len(movie_links)} movie links in {category_name}")
            
            successful_count = 0
            
            for i, movie_link in enumerate(movie_links, 1):
                print(f"  [{i}/{len(movie_links)}] Processing...")
                
                movie_id = self.get_movie_id_from_url(movie_link)
                if movie_id:
                    movie_data = self.get_complete_movie_data_from_api(movie_id)
                    if movie_data:
                        movie_data['category'] = category_name
                        movie_data['source_url'] = movie_link
                        all_movies_data.append(movie_data)
                        successful_count += 1
                        print(f"     Added: {movie_data['title']}")
                    else:
                        print(f"     Failed to get API data")
                else:
                    print(f"     Invalid movie URL")
                
                if i % 10 == 0:
                    time.sleep(1)
            
            print(f"Successfully processed {successful_count}/{len(movie_links)} movies from {category_name}")
        
        try:
            self.driver.quit()
        except Exception:
            pass
        
        if all_movies_data:
            df = pd.DataFrame(all_movies_data)
            return df
        else:
            print("No movie data was collected!")
            return pd.DataFrame()

def main():
    """Main function for successful pagination scraping"""

    API_KEY = os.getenv('TMDB_API_KEY', "eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiI3YmI0MmUyYTc5YWYyNDA1MzU3MDhkODk3MDk0YTBjYSIsIm5iZiI6MTc2Mzc2NDg2NS4xNTI5OTk5LCJzdWIiOiI2OTIwZWE4MTM5YmEwOTMwYjA1ZDY1YWYiLCJzY29wZXMiOlsiYXBpX3JlYWQiXSwidmVyc2lvbiI6MX0.BvjwivRMjFRHXNu_iEQ8BgyEq5wxoDXQefpLXNDirFU")
    
    print("Starting TMDB SUCCESSFUL Web Scraping")
    print("Using WORKING pagination approach")
    print("URL Pattern: https://www.themoviedb.org/movie?page={}")
    print("Expected: 200-1000+ movies across all categories")
    print("This will get you ALL the movies...")
    
    # if running on CI, set headless True via env var CI or HEADLESS
    headless_env = os.getenv('HEADLESS')
    ci_env = os.getenv('CI')
    headless_flag = True if headless_env == '1' or headless_env == 'true' or ci_env else False

    scraper = TMDBSuccessfulScraper(api_key=API_KEY, headless=headless_flag)
    
    start_time = time.time()
    df = scraper.scrape_all_categories_complete(pages_per_category=20)
    end_time = time.time()
    execution_time = end_time - start_time
    execution_minutes = execution_time / 60
    
    if not df.empty:
        output_file = "data/tmdb_movies.csv"  
        os.makedirs('data', exist_ok=True)  
        df.to_csv(output_file, index=False, encoding='utf-8')
        
        print(f"\n{'='*70}")
        print(f"HUGE DATASET CREATED SUCCESSFULLY!")
        print(f"{'='*70}")
        print(f"Total movies scraped: {len(df)}")
        print(f"Execution time: {execution_minutes:.2f} minutes")
        print(f"Data saved to: {output_file}")
        
        print(f"\nCATEGORY BREAKDOWN:")
        for category in df['category'].unique():
            count = len(df[df['category'] == category])
            print(f"   {category}: {count} movies")
        
        print(f"\nCOMPLETE DATA QUALITY:")
        fields_to_check = [
            'budget', 'revenue', 'cast', 'director', 'keywords',
            'production_companies', 'release_date', 'runtime', 'vote_average'
        ]
        
        for field in fields_to_check:
            if field in df.columns:
                if field in ['cast', 'director', 'keywords', 'production_companies', 'genres']:
                    count = df[field].apply(lambda x: len(x) if x and x != 'N/A' and x != [] else 0).gt(0).sum()
                else:
                    count = df[field].apply(lambda x: x not in ['N/A', None, ''] and x != 0).sum()
                print(f"   {field.replace('_', ' ').title()}: {count}/{len(df)} movies")
        
        print("-" * 100)
        sample = df.head(3)
        for idx, row in sample.iterrows():
            print(f"{row['title']} ({row.get('release_date', 'N/A')})")
            print(f"   {row['category']} | {row.get('vote_average', 'N/A')} | {row.get('runtime', 'N/A')}min")
            print(f"   Budget: ${row.get('budget', 'N/A'):,} | Revenue: ${row.get('revenue', 'N/A'):,}")
            print(f"   Genres: {', '.join(row['genres']) if row.get('genres') else 'N/A'}")
            print(f"   Director: {', '.join(row['director']) if row.get('director') else 'N/A'}")
            print()
        
    else:
        print("No data was collected!")

if __name__ == "__main__":
    main()