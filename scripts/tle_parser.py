import pandas as pd
import numpy as np
from skyfield.api import EarthSatellite, load
import os

def parse_tle_data():
    """Parse TLE data and extract orbital parameters"""
    
    debris_list = []
    
    try:
        with open('data/debris_tle.txt') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print("TLE data file not found. Please run space_track.py first.")
        return None
    
    print(f"Parsing {len(lines)} lines of TLE data...")
    
    # Parse every 3 lines (name + line1 + line2)
    for i in range(0, len(lines), 3):
        if i + 2 >= len(lines):
            break
            
        try:
            name = lines[i].strip()
            line1 = lines[i+1].strip()
            line2 = lines[i+2].strip()
            # Always extract norad_id as string
            norad_id = line2[2:7].strip()
            satellite = EarthSatellite(line1, line2, name)
            debris_list.append({
                'norad_id': norad_id,  # Always string
                'name': name,
                'tle_line1': line1,
                'tle_line2': line2,
                'sat_obj': satellite  # Skyfield object
            })
        except Exception as e:
            print(f"Error parsing TLE entry {i//3}: {e}")
            continue
    
    debris_df = pd.DataFrame(debris_list)
    debris_df['norad_id'] = debris_df['norad_id'].astype(str)  # Ensure string type
    print(f"Successfully parsed {len(debris_df)} satellite objects")
    
    return debris_df

def extract_orbital_elements(debris_df):
    """Extract orbital elements from satellite objects"""
    
    ts = load.timescale()
    
    def get_orbital_elements(row):
        try:
            sat = row['sat_obj']
            t = ts.now()  # Current time
            geocentric = sat.at(t)
            
            # Extract orbital elements
            return pd.Series({
                'semi_major_axis': sat.model.no_kozai,  # Semi-major axis (Earth radii)
                'eccentricity': sat.model.ecco,
                'inclination': sat.model.inclo,
                'raan': sat.model.nodeo,  # Right Ascension of Ascending Node
                'arg_perigee': sat.model.argpo,
                'mean_motion': sat.model.no  # Revolutions/day
            })
        except Exception as e:
            print(f"Error extracting orbital elements for {row['norad_id']}: {e}")
            return pd.Series({
                'semi_major_axis': np.nan,
                'eccentricity': np.nan,
                'inclination': np.nan,
                'raan': np.nan,
                'arg_perigee': np.nan,
                'mean_motion': np.nan
            })
    
    print("Extracting orbital elements...")
    orbital_elements = debris_df.apply(get_orbital_elements, axis=1)
    debris_df = debris_df.join(orbital_elements)
    
    # Save to CSV (without sat_obj column which can't be serialized)
    csv_df = debris_df.drop(columns=['sat_obj'])
    csv_df['norad_id'] = csv_df['norad_id'].astype(str)  # Ensure string type in CSV
    csv_df.to_csv('data/orbital_params.csv', index=False)
    print("Orbital parameters saved to data/orbital_params.csv")
    
    return debris_df

if __name__ == "__main__":
    # Parse TLE data
    debris_df = parse_tle_data()
    
    if debris_df is not None:
        # Extract orbital elements
        debris_df = extract_orbital_elements(debris_df)
        print(f"Final dataset contains {len(debris_df)} objects") 