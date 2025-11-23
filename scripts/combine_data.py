import pandas as pd
import logging
from datetime import datetime
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def combine_movie_data():
    """Combine TMDB and IMDb data into a single dataset"""
    try:
       
        tmdb_df = pd.read_csv('data/tmdb_movies.csv')
        imdb_df = pd.read_csv('data/imdb_movies.csv')
        
        logger.info(f"TMDB data: {len(tmdb_df)} movies")
        logger.info(f"IMDb data: {len(imdb_df)} movies")
        

        tmdb_df['source'] = 'TMDB'
        imdb_df['source'] = 'IMDb'
        
       
        combined_df = pd.concat([tmdb_df, imdb_df], ignore_index=True)
        
       
        combined_df['combined_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
      
        combined_df.to_csv('data/combined_movies.csv', index=False)
        
        logger.info(f"Combined data: {len(combined_df)} total movies")
        logger.info(f"Sources: {combined_df['source'].value_counts().to_dict()}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error combining data: {e}")
        return False

if __name__ == "__main__":
    combine_movie_data()