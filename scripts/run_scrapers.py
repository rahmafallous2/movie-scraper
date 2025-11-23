import sys
import os
import logging
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scrapers'))


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_all_scrapers():
    """Run both TMDB and IMDb scrapers"""
    try:
  
        logger.info("Starting TMDB scraper...")
        from scrapers.tmdb_scraper import main as tmdb_main
        tmdb_main()
        
       
        logger.info("Starting IMDb scraper...")
        from scrapers.imdb_scraper import main as imdb_main
        imdb_main()
        
        logger.info("Both scrapers completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error running scrapers: {e}")
        return False

if __name__ == "__main__":
    run_all_scrapers()