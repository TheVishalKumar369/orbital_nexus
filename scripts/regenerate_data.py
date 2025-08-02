#!/usr/bin/env python3
"""
Script to regenerate orbital parameters data with improved parsing logic.
This will help reduce the data quality issues and improve model training effectiveness.
"""

import os
import sys
import pandas as pd
from scripts.tle_parser import parse_tle_data, extract_orbital_elements

def main():
    """Regenerate orbital parameters data with improved parsing"""
    
    print("=== Space Debris Tracker - Data Regeneration ===")
    print()
    
    # Check if TLE data exists
    if not os.path.exists('data/debris_tle.txt'):
        print("❌ Error: TLE data file not found!")
        print("Please run the system and download TLE data first through the web interface.")
        sys.exit(1)
    
    print("✅ TLE data file found")
    
    # Parse TLE data
    print("🔄 Parsing TLE data...")
    debris_df = parse_tle_data()
    
    if debris_df is None:
        print("❌ Error: Failed to parse TLE data!")
        sys.exit(1)
    
    print(f"✅ Successfully parsed {len(debris_df)} satellite objects")
    
    # Extract orbital elements with improved error handling
    print("🔄 Extracting orbital elements with improved validation...")
    debris_df = extract_orbital_elements(debris_df)
    
    print(f"✅ Successfully extracted orbital elements for {len(debris_df)} objects")
    
    # Validate the improved data quality
    print("\n=== Data Quality Report ===")
    
    # Check for NaN values in critical columns
    feature_columns = ['semi_major_axis', 'eccentricity', 'inclination', 'mean_motion']
    
    total_objects = len(debris_df)
    valid_objects = 0
    
    for _, row in debris_df.iterrows():
        if all(pd.notna(row[col]) and not pd.isna(row[col]) for col in feature_columns):
            valid_objects += 1
    
    data_quality_percentage = (valid_objects / total_objects) * 100
    
    print(f"📊 Total objects: {total_objects}")
    print(f"📊 Valid objects (no NaN): {valid_objects}")
    print(f"📊 Data quality: {data_quality_percentage:.1f}%")
    
    if data_quality_percentage > 90:
        print("✅ Excellent data quality!")
    elif data_quality_percentage > 80:
        print("⚠️  Good data quality - some improvements made")
    else:
        print("❌ Data quality still needs improvement")
    
    # Show sample statistics
    print("\n=== Sample Statistics ===")
    print("Semi-major axis range:", debris_df['semi_major_axis'].min(), "to", debris_df['semi_major_axis'].max())
    print("Eccentricity range:", debris_df['eccentricity'].min(), "to", debris_df['eccentricity'].max())
    print("Mean motion range:", debris_df['mean_motion'].min(), "to", debris_df['mean_motion'].max())
    
    print("\n✅ Data regeneration complete!")
    print("🚀 You can now train the model with improved data quality.")
    print()
    print("Next steps:")
    print("1. Go to the web interface")
    print("2. Click 'Train Model' to retrain with improved data")
    print("3. The model should now use more training samples and perform better")

if __name__ == "__main__":
    main()
