import os
import json
import time
from spacetrack import SpaceTrackClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

CACHE_FILE = 'data/last_tle_download.json'
TLE_DOWNLOAD_INTERVAL = 5400  # 1.5 hours in seconds (for extra safety)


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

    # Initialize Space-Track client
    st = SpaceTrackClient(
        identity=os.getenv('SPACETRACK_USERNAME'),
        password=os.getenv('SPACETRACK_PASSWORD')
    )

    try:
        # Download Two-Line Element (TLE) data for all debris
        print("Downloading TLE data from Space-Track.org...")
        tle_data = st.tle(
            iter_lines=True,
            # norad_cat_id=range(1, 50000),  # All objects (commented out: caused URL too long error)
            epoch='>now-7',                # Last 30 days
            format='tle'
        )

        # Save to file
        with open('data/debris_tle.txt', 'w') as f:
            for line in tle_data:
                f.write(line + '\n')

        # Update cache
        with open(CACHE_FILE, 'w') as f:
            json.dump({'last_download': now}, f)

        print(f"Successfully downloaded TLE data to data/debris_tle.txt")

    except Exception as e:
        if '429' in str(e):
            print("Error: Rate limit exceeded (HTTP 429). Please wait and try again later.")
        else:
            print(f"Error downloading TLE data: {e}")
        return False

    return True

if __name__ == "__main__":
    download_tle_data() 