import os
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import pandas as pd
import time
import re
import json
from urllib.parse import quote, urljoin
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IMDbGenreFinder:
    def __init__(self):
        self.driver = None
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        })
        self.base_url = "https://www.imdb.com"
        self.movies_scraped = 0
        
    def setup_driver(self):
        """Setup Selenium WebDriver"""
        chrome_options = Options()
        # chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        
        self.driver = webdriver.Chrome(options=chrome_options)
        self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        
    def close_driver(self):
        if self.driver:
            self.driver.quit()

    def extract_genres_advanced(self):
        """Extract genres from ipc-chip__text spans"""
        try:
            genres = []
            
       
            try:
                genre_spans = self.driver.find_elements(By.CSS_SELECTOR, 'span.ipc-chip__text')
                for span in genre_spans:
                    genre = span.text.strip()
                    if genre and genre not in genres:
                        genres.append(genre)
                logger.info(f"Found {len(genres)} genres from ipc-chip__text")
            except Exception as e:
                logger.info(f"ipc-chip__text method failed: {e}")
            
        
            if not genres:
                try:
                    genre_links = self.driver.find_elements(By.CSS_SELECTOR, 'a[href*="genres="], a[href*="/search/title/?genres="]')
                    for link in genre_links:
                        genre = link.text.strip()
                        if genre and genre not in ['Genres', 'Genre', 'More'] and genre not in genres:
                            genres.append(genre)
                    logger.info(f"Fallback: Found {len(genres)} genres from links")
                except Exception as e:
                    logger.info(f"Fallback method failed: {e}")
            
            return list(set(genres))  
            
        except Exception as e:
            logger.error(f"Genre extraction failed: {e}")
            return []

    def get_movie_urls_from_category(self, category_url, max_movies=100):
        """Get movie URLs from a specific category - INCREASED LIMIT"""
        if not self.driver:
            self.setup_driver()
            
        logger.info(f"Getting up to {max_movies} movies from category: {category_url}")
        
        try:
            self.driver.get(category_url)
            time.sleep(3)
            
     
            WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            movie_urls = []
            

            selectors_to_try = [
                'a[href*="/title/tt"][href*="/?ref_=tt_ov_inf"]',  
                'a[href*="/title/tt"][href*="/?ref_=adv_li_tt"]',  
                'a[href*="/title/tt"][href*="/?ref_=tt_sims_tt"]',  
                'a.ipc-poster-card__title',  
                'h3.ipc-title__text a',  
                '.lister-item-header a',  
                '.titleColumn a',  
                'a[href*="/title/tt"]'  
            ]
            
            for selector in selectors_to_try:
                try:
                    movie_elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    logger.info(f"🔍 Selector '{selector}' found {len(movie_elements)} elements")
                    
                    for element in movie_elements:
                        if len(movie_urls) >= max_movies:
                            break
                            
                        href = element.get_attribute('href')
                        if href and '/title/tt' in href:
                            
                            clean_url = href.split('?')[0]  
                            if clean_url.endswith('/') and clean_url not in movie_urls:
                                movie_urls.append(clean_url)
                    
                    if len(movie_urls) >= max_movies:
                        break
                        
                except Exception as e:
                    logger.info(f"Selector '{selector}' failed: {e}")
                    continue
            
           
            if not movie_urls:
                logger.info("🔄 Trying generic movie discovery...")
                all_links = self.driver.find_elements(By.TAG_NAME, 'a')
                for link in all_links:
                    if len(movie_urls) >= max_movies:
                        break
                    href = link.get_attribute('href')
                    if href and '/title/tt' in href and '/?ref_=' in href:
                        clean_url = href.split('?')[0]
                        if clean_url.endswith('/') and clean_url not in movie_urls:
                            movie_urls.append(clean_url)
            
            logger.info(f"📋 Found {len(movie_urls)} movies in category")
            return movie_urls[:max_movies]
            
        except Exception as e:
            logger.error(f"Error getting movies from category {category_url}: {e}")
            return []

    def get_all_category_urls(self):
        """Get URLs for all movie categories - EXPANDED LIST"""
        categories = {
            "Popular Movies": "https://www.imdb.com/chart/moviemeter/",
            "Top Rated Movies": "https://www.imdb.com/chart/top/",
            "Box Office": "https://www.imdb.com/chart/boxoffice/",
            "Coming Soon": "https://www.imdb.com/chart/comingsoon/",
            "Action": "https://www.imdb.com/search/title/?genres=action&title_type=feature",
            "Adventure": "https://www.imdb.com/search/title/?genres=adventure&title_type=feature",
            "Animation": "https://www.imdb.com/search/title/?genres=animation&title_type=feature",
            "Comedy": "https://www.imdb.com/search/title/?genres=comedy&title_type=feature",
            "Crime": "https://www.imdb.com/search/title/?genres=crime&title_type=feature",
            "Documentary": "https://www.imdb.com/search/title/?genres=documentary&title_type=feature",
            "Drama": "https://www.imdb.com/search/title/?genres=drama&title_type=feature",
            "Fantasy": "https://www.imdb.com/search/title/?genres=fantasy&title_type=feature",
            "Horror": "https://www.imdb.com/search/title/?genres=horror&title_type=feature",
            "Mystery": "https://www.imdb.com/search/title/?genres=mystery&title_type=feature",
            "Romance": "https://www.imdb.com/search/title/?genres=romance&title_type=feature",
            "Sci-Fi": "https://www.imdb.com/search/title/?genres=sci-fi&title_type=feature",
            "Thriller": "https://www.imdb.com/search/title/?genres=thriller&title_type=feature",
         
            "Family": "https://www.imdb.com/search/title/?genres=family&title_type=feature",
            "History": "https://www.imdb.com/search/title/?genres=history&title_type=feature",
            "Music": "https://www.imdb.com/search/title/?genres=music&title_type=feature",
            "War": "https://www.imdb.com/search/title/?genres=war&title_type=feature",
            "Western": "https://www.imdb.com/search/title/?genres=western&title_type=feature",
            "Sport": "https://www.imdb.com/search/title/?genres=sport&title_type=feature"
        }
        
        return categories

    def get_movie_data_fast(self, movie_url):
        """Fast movie data extraction - focused on essential fields"""
        if not self.driver:
            self.setup_driver()
            
        imdb_id = self.extract_imdb_id(movie_url)
        if not imdb_id:
            return None
            
        logger.info(f"Scraping movie {self.movies_scraped + 1}: {imdb_id}")
        
        try:
            self.driver.get(movie_url)
            time.sleep(2) 
            
            try:
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, "h1"))
                )
            except:
                logger.warning(f"Page may not have loaded properly for {movie_url}")
            
           
            movie_data = {
                'index': self.movies_scraped + 1,
                'title': self.extract_title(),
                'genres': self.extract_genres_advanced(),
                'id': imdb_id,
                'release_date': self.extract_release_date(),
                'vote_average': self.extract_vote_average(),
                'vote_count': self.extract_vote_count(),
                'runtime': self.extract_runtime(),
                'overview': self.extract_overview(),
                'director': self.extract_director(),
                'cast': self.extract_cast()[:5], 
                'budget': self.extract_budget(),
                'revenue': self.extract_revenue(),
                'original_language': self.extract_original_language(),
                'production_companies': self.extract_production_companies(),
                'source_url': movie_url
            }
            
            if movie_data.get('title'):
                self.movies_scraped += 1
                logger.info(f"Success: {movie_data['title']} - Genres: {movie_data['genres']}")
                return movie_data
            else:
                logger.warning(f"Failed to extract title for {movie_url}")
                return None
                
        except Exception as e:
            logger.error(f"Error scraping {movie_url}: {e}")
            return None

    def extract_all_fields(self, imdb_id, movie_url):
        """Extract all fields - comprehensive version"""
        movie_data = {'index': self.movies_scraped + 1}
        
        
        movie_data['title'] = self.extract_title()
        
   
        movie_data['genres'] = self.extract_genres_advanced()
        
        
        movie_data['budget'] = self.extract_budget()
        movie_data['homepage'] = self.extract_homepage()
        movie_data['id'] = imdb_id
        movie_data['keywords'] = self.extract_keywords()
        movie_data['original_language'] = self.extract_original_language()
        movie_data['original_title'] = self.extract_original_title()
        movie_data['overview'] = self.extract_overview()
        movie_data['popularity'] = self.extract_popularity()
        movie_data['production_companies'] = self.extract_production_companies()
        movie_data['production_countries'] = self.extract_production_countries()
        movie_data['release_date'] = self.extract_release_date()
        movie_data['revenue'] = self.extract_revenue()
        movie_data['runtime'] = self.extract_runtime()
        movie_data['spoken_languages'] = self.extract_spoken_languages()
        movie_data['status'] = self.extract_status()
        movie_data['tagline'] = self.extract_tagline()
        movie_data['vote_average'] = self.extract_vote_average()
        movie_data['vote_count'] = self.extract_vote_count()
        movie_data['cast'] = self.extract_cast()
        movie_data['director'] = self.extract_director()
        movie_data['crew'] = self.extract_crew()
        movie_data['category'] = "Feature Film"
        movie_data['source_url'] = movie_url
        
        return movie_data

    def extract_imdb_id(self, url):
        """Extract IMDb ID from URL"""
        match = re.search(r'/title/(tt\d+)', url)
        return match.group(1) if match else None

    def extract_title(self):
        try:
            selectors = ['h1', 'h1[data-testid*="title"]', '.title_wrapper h1']
            for selector in selectors:
                try:
                    element = self.driver.find_element(By.CSS_SELECTOR, selector)
                    title = element.text.strip()
                    if title and len(title) > 0:
                        return title
                except:
                    continue
            return ""
        except:
            return ""

    def extract_director(self):
        """Extract director using JavaScript"""
        try:
            script = """
            var directors = [];
            var directorSections = document.querySelectorAll('[data-testid="title-pc-principal-credit"]');
            directorSections.forEach(function(section) {
                if (section.textContent.includes('Director')) {
                    var links = section.querySelectorAll('a');
                    links.forEach(function(link) {
                        var name = link.textContent.trim();
                        if (name && !directors.includes(name)) {
                            directors.push(name);
                        }
                    });
                }
            });
            return directors;
            """
            return self.driver.execute_script(script) or []
        except:
            return []

   
    def extract_budget(self):
        try:
            budget_selectors = [
                'li[data-testid="title-boxoffice-budget"]',
                '//span[contains(text(), "Budget")]/following-sibling::span'
            ]
            
            for selector in budget_selectors:
                try:
                    if selector.startswith('//'):
                        element = self.driver.find_element(By.XPATH, selector)
                    else:
                        element = self.driver.find_element(By.CSS_SELECTOR, selector)
                    
                    budget_text = element.text
                    return self.parse_currency(budget_text)
                except:
                    continue
            return 0
        except:
            return 0

    def extract_homepage(self):
        try:
            homepage_selectors = [
                'a[data-testid="hero-title-block__homepage-link"]',
                '//a[contains(text(), "Homepage")]'
            ]
            
            for selector in homepage_selectors:
                try:
                    if selector.startswith('//'):
                        element = self.driver.find_element(By.XPATH, selector)
                    else:
                        element = self.driver.find_element(By.CSS_SELECTOR, selector)
                    href = element.get_attribute('href')
                    if href and '://' in href:
                        return href
                except:
                    continue
            return ""
        except:
            return ""

    def extract_keywords(self):
        try:
            keywords = []
            keyword_elements = self.driver.find_elements(By.CSS_SELECTOR, 'a[href*="/keyword/"]')
            for element in keyword_elements:
                keyword = element.text.strip()
                if keyword and len(keyword) > 1 and keyword not in keywords:
                    keywords.append(keyword)
            return keywords[:15]
        except:
            return []

    def extract_original_language(self):
        try:
            language_selectors = [
                'li[data-testid="title-details-languages"] a',
                '//a[contains(@href, "primary_language")]'
            ]
            
            for selector in language_selectors:
                try:
                    if selector.startswith('//'):
                        element = self.driver.find_element(By.XPATH, selector)
                    else:
                        element = self.driver.find_element(By.CSS_SELECTOR, selector)
                    lang = element.text.strip()
                    if lang:
                        return lang
                except:
                    continue
            return "English"
        except:
            return "English"

    def extract_original_title(self):
        return self.extract_title()

    def extract_overview(self):
        try:
            plot_selectors = [
                '[data-testid="plot"] span',
                '[data-testid="plot-xl"]',
                '[data-testid="plot-l"]'
            ]
            
            for selector in plot_selectors:
                try:
                    element = self.driver.find_element(By.CSS_SELECTOR, selector)
                    plot = element.text.strip()
                    if plot:
                        return plot
                except:
                    continue
            return ""
        except:
            return ""

    def extract_popularity(self):
        try:
            vote_count = self.extract_vote_count()
            rating = self.extract_vote_average()
            return round((vote_count * rating) / 1000, 2) if vote_count > 0 else 0.0
        except:
            return 0.0

    def extract_production_companies(self):
        try:
            companies = []
            company_selectors = [
                'li[data-testid="title-details-companies"] a',
                '//a[contains(@href, "/search/title/?companies=")]'
            ]
            
            for selector in company_selectors:
                try:
                    if selector.startswith('//'):
                        elements = self.driver.find_elements(By.XPATH, selector)
                    else:
                        elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    
                    for element in elements:
                        company = element.text.strip()
                        if company and company not in companies:
                            companies.append(company)
                    if companies:
                        break
                except:
                    continue
            return companies
        except:
            return []

    def extract_production_countries(self):
        try:
            countries = []
            country_selectors = [
                'li[data-testid="title-details-origin"] a',
                '//a[contains(@href, "country_of_origin")]'
            ]
            
            for selector in country_selectors:
                try:
                    if selector.startswith('//'):
                        elements = self.driver.find_elements(By.XPATH, selector)
                    else:
                        elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    
                    for element in elements:
                        country = element.text.strip()
                        if country and country not in countries:
                            countries.append(country)
                    if countries:
                        break
                except:
                    continue
            return countries
        except:
            return []

    def extract_release_date(self):
        try:
            date_selectors = [
                'a[href*="/releaseinfo"]',
                '[data-testid="title-details-releasedate"] a'
            ]
            
            for selector in date_selectors:
                try:
                    element = self.driver.find_element(By.CSS_SELECTOR, selector)
                    date = element.text.strip()
                    if date:
                        return date
                except:
                    continue
            return ""
        except:
            return ""

    def extract_revenue(self):
        try:
            revenue_selectors = [
                'li[data-testid="title-boxoffice-cumulativeworldwidegross"]',
                '//span[contains(text(), "Cumulative Worldwide Gross")]/following-sibling::span'
            ]
            
            for selector in revenue_selectors:
                try:
                    if selector.startswith('//'):
                        element = self.driver.find_element(By.XPATH, selector)
                    else:
                        element = self.driver.find_element(By.CSS_SELECTOR, selector)
                    
                    revenue_text = element.text
                    return self.parse_currency(revenue_text)
                except:
                    continue
            return 0
        except:
            return 0

    def extract_runtime(self):
        try:
            runtime_selectors = [
                'li[data-testid="title-techspec_runtime"]',
                '//span[contains(text(), "Runtime")]/following-sibling::span'
            ]
            
            for selector in runtime_selectors:
                try:
                    if selector.startswith('//'):
                        element = self.driver.find_element(By.XPATH, selector)
                    else:
                        element = self.driver.find_element(By.CSS_SELECTOR, selector)
                    
                    runtime_text = element.text
                    return self.parse_runtime(runtime_text)
                except:
                    continue
            return 0
        except:
            return 0

    def extract_spoken_languages(self):
        try:
            languages = []
            try:
                lang_elements = self.driver.find_elements(By.CSS_SELECTOR, 'li[data-testid="title-details-languages"] a')
                for element in lang_elements:
                    lang = element.text.strip()
                    if lang and lang not in languages:
                        languages.append(lang)
            except:
                pass
            return languages
        except:
            return []

    def extract_status(self):
        return "Released"

    def extract_tagline(self):
        try:
            tagline_selectors = [
                '[data-testid="storyline-taglines"]',
                '.tagline'
            ]
            
            for selector in tagline_selectors:
                try:
                    element = self.driver.find_element(By.CSS_SELECTOR, selector)
                    tagline = element.text.strip()
                    if tagline:
                        return tagline
                except:
                    continue
            return ""
        except:
            return ""

    def extract_vote_average(self):
        try:
            rating_selectors = [
                '[data-testid="hero-rating-bar__aggregate-rating__score"]',
                '.ratingValue strong'
            ]
            
            for selector in rating_selectors:
                try:
                    element = self.driver.find_element(By.CSS_SELECTOR, selector)
                    rating_text = element.text.strip()
                    if '/' in rating_text:
                        return float(rating_text.split('/')[0].strip())
                    else:
                        return float(rating_text)
                except:
                    continue
            return 0.0
        except:
            return 0.0

    def extract_vote_count(self):
        try:
            try:
                rating_element = self.driver.find_element(By.CSS_SELECTOR, '[data-testid="hero-rating-bar__aggregate-rating__score"]')
                parent = rating_element.find_element(By.XPATH, "./..")
                siblings = parent.find_elements(By.XPATH, "./*")
                
                for sibling in siblings:
                    text = sibling.text
                    if any(x in text.lower() for x in ['votes', 'ratings']):
                        numbers = re.findall(r'[\d,]+', text)
                        if numbers:
                            return int(numbers[0].replace(',', ''))
            except:
                pass
            
            return 0
        except:
            return 0

    def extract_cast(self):
        try:
            cast = []
            cast_selectors = [
                '[data-testid="title-cast-item"]',
                '.cast_list tr'
            ]
            
            for selector in cast_selectors:
                try:
                    cast_elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    
                    for element in cast_elements[:15]:
                        try:
                            name_element = element.find_element(By.CSS_SELECTOR, '[data-testid="title-cast-item__actor"]')
                            character_element = element.find_element(By.CSS_SELECTOR, '.character, [class*="character"]')
                            
                            name = name_element.text.strip()
                            character = character_element.text.strip()
                            
                            if name and character:
                                cast.append(f"{name} as {character}")
                        except:
                            text = element.text.strip()
                            if ' as ' in text and len(text) < 100:
                                cast.append(text)
                    if cast:
                        break
                except:
                    continue
            return cast
        except:
            return []

    def extract_crew(self):
        try:
            crew = []
            crew_roles = ['Writer', 'Producer', 'Screenplay']
            
            for role in crew_roles:
                try:
                    elements = self.driver.find_elements(By.XPATH, f'//a[contains(@href, "/name/")][.//span[contains(text(), "{role}")]]')
                    for element in elements:
                        person = element.text.strip()
                        if person and person not in crew:
                            crew.append(f"{person} ({role})")
                except:
                    continue
            
            return crew[:8]
        except:
            return []

    def parse_currency(self, text):
        if not text:
            return 0
        numbers = re.findall(r'[\d,.]+', text)
        if not numbers:
            return 0
        amount = float(numbers[0].replace(',', ''))
        if 'million' in text.lower():
            amount *= 1000000
        elif 'billion' in text.lower():
            amount *= 1000000000
        return int(amount)

    def parse_runtime(self, text):
        if not text:
            return 0
        hours = re.findall(r'(\d+)h', text)
        minutes = re.findall(r'(\d+)m', text)
        total = 0
        if hours:
            total += int(hours[0]) * 60
        if minutes:
            total += int(minutes[0])
        return total

    def scrape_massive_movie_collection(self, movies_per_category=50, max_total_movies=1000):
        """MASSIVE scraping - get ALL the movies!"""
        all_movies_data = []
        all_movie_urls = set()
        
        categories = self.get_all_category_urls()
        logger.info(f"MASSIVE SCRAPING: Starting to scrape from {len(categories)} categories")
        logger.info(f"TARGET: {max_total_movies} movies total")
        
        for category_name, category_url in categories.items():
            if len(all_movies_data) >= max_total_movies:
                logger.info(f"REACHED TARGET: {max_total_movies} movies!")
                break
                
            remaining = max_total_movies - len(all_movies_data)
            current_target = min(movies_per_category, remaining)
            
            logger.info(f"Processing category: {category_name} (target: {current_target} movies)")
            
            movie_urls = self.get_movie_urls_from_category(category_url, current_target)
            
            for i, movie_url in enumerate(movie_urls):
                if len(all_movies_data) >= max_total_movies:
                    break
                    
                if movie_url not in all_movie_urls:
                    all_movie_urls.add(movie_url)
                    
                   
                    movie_data = self.get_movie_data_fast(movie_url)
                    if movie_data:
                        movie_data['category_name'] = category_name
                        all_movies_data.append(movie_data)
                        
                     
                        if len(all_movies_data) % 10 == 0:
                            logger.info(f"PROGRESS: {len(all_movies_data)}/{max_total_movies} movies collected")
                    

                    time.sleep(1)
                    
                 
                    if len(all_movies_data) % 20 == 0:
                        self.save_progress(all_movies_data)
        
        return all_movies_data
    
    def save_progress(self, movies_data):
        if movies_data:
            df = pd.DataFrame(movies_data)
            os.makedirs('data', exist_ok=True)  
            df.to_csv('data/imdb_movies.csv', index=False) 
            logger.info(f"PROGRESS SAVED: {len(movies_data)} movies")

    def save_final_data(self, movies_data):
        if movies_data:
            df = pd.DataFrame(movies_data)
            os.makedirs('data', exist_ok=True)
            
         
            required_fields = [
                'index', 'title', 'genres', 'id', 'release_date', 'vote_average', 
                'vote_count', 'runtime', 'overview', 'director', 'cast', 'budget',
                'revenue', 'original_language', 'production_companies', 'category_name', 'source_url'
            ]
            
            for field in required_fields:
                if field not in df.columns:
                    df[field] = ""
            
            df = df[required_fields]
            df.to_csv('data/imdb_movies.csv', index=False)  
            logger.info(f"COMPLETE: Saved {len(movies_data)} movies to data/imdb_movies.csv")
            
            print("\n📊 MASSIVE DATA COLLECTION SUMMARY:")
            print(f"🎬 TOTAL MOVIES: {len(movies_data)}")
            print(f"📈 GENRES COLLECTED: Sample - {movies_data[0]['genres'] if movies_data else 'None'}")
            print(f"💾 SAVED TO: imdb_movies_MASSIVE_final.csv")

def main():
    scraper = IMDbGenreFinder()
    
    try:
        logger.info("🚀 STARTING MASSIVE IMDb SCRAPER - 1000 MOVIES TARGET!")
        
        
        all_movies_data = scraper.scrape_massive_movie_collection(
            movies_per_category=60,  
            max_total_movies=1000      
        )
        
        if all_movies_data:
            scraper.save_final_data(all_movies_data)
            logger.info(f"MASSIVE SUCCESS: Scraped {len(all_movies_data)} movies!")
        else:
            logger.error("No data was collected!")
            
    except Exception as e:
        logger.error(f"Main error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        scraper.close_driver()

if __name__ == "__main__":
    main() 