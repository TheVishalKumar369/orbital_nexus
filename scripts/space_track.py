import os
import json
import time
import requests
from datetime import datetime, timedelta
from spacetrack import SpaceTrackClient
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CACHE_FILE = 'data/last_tle_download.json'
RATE_LIMIT_FILE = 'data/rate_limit_cache.json'
TLE_DOWNLOAD_INTERVAL = 3600  # 1 hour in seconds
QUERY_RATE_LIMIT = 20  # 20 queries per minute
QUERY_INTERVAL = 60  # 1 minute in seconds
RETRY_ATTEMPTS = 3
RETRY_DELAY = 30  # seconds


def check_rate_limit():
    """Check if we're within the rate limit of 20 queries per minute"""
    now = time.time()
    
    # Load rate limit cache
    rate_cache = {'queries': [], 'last_reset': now}
    if os.path.exists(RATE_LIMIT_FILE):
        try:
            with open(RATE_LIMIT_FILE, 'r') as f:
                rate_cache = json.load(f)
        except Exception:
            pass
    
    # Remove queries older than 1 minute
    minute_ago = now - QUERY_INTERVAL
    rate_cache['queries'] = [q for q in rate_cache['queries'] if q > minute_ago]
    
    # Check if we can make another query
    if len(rate_cache['queries']) >= QUERY_RATE_LIMIT:
        oldest_query = min(rate_cache['queries'])
        wait_time = QUERY_INTERVAL - (now - oldest_query)
        logger.warning(f"Rate limit reached. Waiting {wait_time:.1f} seconds...")
        time.sleep(wait_time + 1)  # Add 1 second buffer
        return check_rate_limit()  # Recursive check
    
    # Record this query
    rate_cache['queries'].append(now)
    
    # Save rate limit cache
    with open(RATE_LIMIT_FILE, 'w') as f:
        json.dump(rate_cache, f)
    
    return True


def validate_credentials():
    """Validate Space-Track credentials"""
    username = os.getenv('SPACETRACK_USERNAME')
    password = os.getenv('SPACETRACK_PASSWORD')
    
    if not username or not password:
        logger.error("Space-Track credentials not found. Please set SPACETRACK_USERNAME and SPACETRACK_PASSWORD environment variables.")
        return False
    
    return True


def download_with_retry(st, **kwargs):
    """Download data with retry logic and rate limiting"""
    for attempt in range(RETRY_ATTEMPTS):
        try:
            # Check rate limit before making request
            check_rate_limit()
            
            logger.info(f"Attempt {attempt + 1}/{RETRY_ATTEMPTS}: Downloading TLE data...")
            
            # Make the request
            tle_data = st.tle(iter_lines=True, **kwargs)
            chunked_data = []
            for line in tle_data:
                chunked_data.append(line)
            return chunked_data
            
        except Exception as e:
            error_msg = str(e).lower()
            
            if '429' in error_msg or 'rate limit' in error_msg:
                wait_time = RETRY_DELAY * (attempt + 1)
                logger.warning(f"Rate limit hit. Waiting {wait_time} seconds before retry {attempt + 1}/{RETRY_ATTEMPTS}...")
                time.sleep(wait_time)
                
            elif '401' in error_msg or 'unauthorized' in error_msg:
                logger.error("Authentication failed. Please check your Space-Track credentials.")
                raise
                
            elif '403' in error_msg or 'forbidden' in error_msg:
                logger.error("Access forbidden. Your account may not have permission to access this data.")
                raise
                
            elif attempt == RETRY_ATTEMPTS - 1:
                logger.error(f"Final attempt failed: {e}")
                raise
            else:
                wait_time = RETRY_DELAY * (attempt + 1)
                logger.warning(f"Request failed: {e}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
    
    raise Exception("All retry attempts exhausted")


def download_tle_data():
    """Download TLE data from Space-Track.org with rate limiting and caching"""
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)

    # Check last download time
    last_download = 0
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                cache = json.load(f)
                last_download = cache.get('last_download', 0)
        except Exception:
            last_download = 0

    now = int(time.time())
    wait_time = TLE_DOWNLOAD_INTERVAL - (now - last_download)
    if wait_time > 0:
        print(f"TLE data was downloaded recently. Please wait {wait_time // 60} min {wait_time % 60} sec before downloading again.")
        return False

    # Validate credentials first
    if not validate_credentials():
        return False

    # Initialize Space-Track client with increased timeout
    st = SpaceTrackClient(
        identity=os.getenv('SPACETRACK_USERNAME'),
        password=os.getenv('SPACETRACK_PASSWORD')
    )
    
    # Configure session timeout (5 minutes)
    if hasattr(st, 'session'):
        st.session.timeout = 300

    try:
        # Use the enhanced download function with retry logic
        logger.info("Starting TLE data download from Space-Track.org...")
        
        # Download parameters for larger dataset
        download_params = {
            'epoch': '>now-7',  # Last 7 days for more data
            'format': 'tle',
            'orderby': 'epoch desc',  # Get newest data first
            'limit': 10000  # Limit to prevent overwhelming requests
        }
        
        tle_lines = download_with_retry(st, **download_params)
        
        if not tle_lines:
            logger.warning("No TLE data received")
            return False
        
        # Save to file with progress indication
        logger.info(f"Saving {len(tle_lines)} lines of TLE data...")
        with open('data/debris_tle.txt', 'w') as f:
            for i, line in enumerate(tle_lines):
                f.write(line + '\n')
                if i > 0 and i % 1000 == 0:
                    logger.info(f"Saved {i} lines...")

        # Update cache
        with open(CACHE_FILE, 'w') as f:
            json.dump({
                'last_download': now,
                'lines_downloaded': len(tle_lines),
                'download_params': download_params
            }, f)

        logger.info(f"Successfully downloaded {len(tle_lines)} lines of TLE data to data/debris_tle.txt")
        print(f"Download completed: {len(tle_lines)} lines saved to data/debris_tle.txt")

    except Exception as e:
        error_msg = str(e)
        if 'timeout' in error_msg.lower():
            logger.error("Download timed out. Try reducing the data range or increasing timeout.")
            print("Error: Download timed out. The dataset might be too large.")
        elif '429' in error_msg:
            logger.error("Rate limit exceeded (HTTP 429). Please wait and try again later.")
            print("Error: Rate limit exceeded (HTTP 429). Please wait and try again later.")
        else:
            logger.error(f"Error downloading TLE data: {e}")
            print(f"Error downloading TLE data: {e}")
        return False

    return True

if __name__ == "__main__":
    download_tle_data() 