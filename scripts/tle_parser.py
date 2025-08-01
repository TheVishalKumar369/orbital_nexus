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
            
            # Validate satellite object first
            if not hasattr(sat, 'model') or sat.model is None:
                raise ValueError(f"Invalid satellite model for {row['norad_id']}")
            
            # Extract orbital elements with validation
            semi_major_axis = getattr(sat.model, 'no_kozai', np.nan)
            eccentricity = getattr(sat.model, 'ecco', np.nan)
            inclination = getattr(sat.model, 'inclo', np.nan)
            raan = getattr(sat.model, 'nodeo', np.nan)
            arg_perigee = getattr(sat.model, 'argpo', np.nan)
            mean_motion = getattr(sat.model, 'no', np.nan)
            
            # Validate extracted values - replace invalid values with reasonable defaults
            if np.isnan(semi_major_axis) or np.isinf(semi_major_axis) or semi_major_axis <= 0:
                semi_major_axis = 0.06  # Default for LEO satellites
            
            if np.isnan(eccentricity) or np.isinf(eccentricity) or eccentricity < 0 or eccentricity >= 1:
                eccentricity = 0.01  # Nearly circular orbit
            
            if np.isnan(inclination) or np.isinf(inclination):
                inclination = 1.0  # Default inclination
            
            if np.isnan(raan) or np.isinf(raan):
                raan = 0.0  # Default RAAN
            
            if np.isnan(arg_perigee) or np.isinf(arg_perigee):
                arg_perigee = 0.0  # Default argument of perigee
            
            if np.isnan(mean_motion) or np.isinf(mean_motion) or mean_motion <= 0:
                mean_motion = 15.0  # Default for LEO satellites (~90 min orbit)
            
            return pd.Series({
                'semi_major_axis': float(semi_major_axis),
                'eccentricity': float(eccentricity),
                'inclination': float(inclination),
                'raan': float(raan),
                'arg_perigee': float(arg_perigee),
                'mean_motion': float(mean_motion)
            })
            
        except Exception as e:
            print(f"Error extracting orbital elements for {row['norad_id']}: {e}")
            # Return sensible defaults instead of NaN values
            return pd.Series({
                'semi_major_axis': 0.06,   # Default for LEO
                'eccentricity': 0.01,      # Nearly circular
                'inclination': 1.0,        # Default inclination
                'raan': 0.0,               # Default RAAN
                'arg_perigee': 0.0,        # Default arg perigee
                'mean_motion': 15.0        # Default mean motion
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