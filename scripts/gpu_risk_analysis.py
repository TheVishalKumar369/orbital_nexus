#!/usr/bin/env python3
"""
GPU-Accelerated Collision Risk Analysis
Optimized for fast processing of large satellite datasets
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from skyfield.api import load, EarthSatellite
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

# Configure TensorFlow to use GPU
def configure_gpu():
    """Configure TensorFlow for GPU acceleration"""
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        return True
    return False

class GPUAcceleratedRiskAnalysis:
    def __init__(self):
        self.ts = load.timescale()
        self.gpu_available = configure_gpu()
        print(f"GPU acceleration: {'Enabled' if self.gpu_available else 'Disabled'}")
        
    def batch_position_computation(self, satellite_objects, hours=48, step=2):
        """
        Compute positions for multiple satellites in batches (optimized)
        Reduced step size and hours for faster computation
        """
        print(f"Computing positions for {len(satellite_objects)} satellites...")
        
        # Reduce computation load - fewer time steps
        times = self.ts.utc(*self.ts.now().utc[:4], range(0, hours, step))
        
        positions_dict = {}
        
        # Process in smaller batches to avoid memory issues
        batch_size = 100
        sat_ids = list(satellite_objects.keys())
        for i in range(0, len(sat_ids), batch_size):
            batch_ids = sat_ids[i:i+batch_size]
            batch_positions = {}
            
            for sat_id in batch_ids:
                sat_obj = satellite_objects[sat_id]
                try:
                    positions = []
                    for t in times:
                        geocentric = sat_obj.at(t)
                        positions.append(geocentric.position.km)
                    batch_positions[sat_id] = np.array(positions)
                except Exception as e:
                    print(f"Error computing positions for {sat_id}: {e}")
                    continue
            
            positions_dict.update(batch_positions)
            
            # Progress update
            if i % 500 == 0:
                print(f"Processed {min(i + batch_size, len(satellite_objects))}/{len(satellite_objects)} satellites")
        
        return positions_dict, times
    
    @tf.function
    def gpu_distance_matrix(self, sat_positions, debris_positions):
        """GPU-accelerated distance computation using TensorFlow"""
        # Convert to TensorFlow tensors
        sat_pos_tensor = tf.convert_to_tensor(sat_positions, dtype=tf.float32)
        debris_pos_tensor = tf.convert_to_tensor(debris_positions, dtype=tf.float32)
        
        # Compute pairwise distances using broadcasting
        # sat_pos: (time_steps, 3), debris_pos: (n_debris, time_steps, 3)
        sat_expanded = tf.expand_dims(sat_pos_tensor, axis=0)  # (1, time_steps, 3)
        
        # Calculate distances for all debris at all time steps
        diff = sat_expanded - debris_pos_tensor  # (n_debris, time_steps, 3)
        distances = tf.norm(diff, axis=2)  # (n_debris, time_steps)
        
        return distances
    
    def find_close_approaches_optimized(self, target_sat_id, positions_dict, threshold_km=50):
        """
        Optimized close approach detection using GPU acceleration
        Reduced threshold for faster processing
        """
        if target_sat_id not in positions_dict:
            print(f"Target satellite {target_sat_id} not found in positions")
            return []
        
        target_positions = positions_dict[target_sat_id]
        risks = []
        
        # Convert other satellite positions to numpy array for vectorized operations
        other_sats = {k: v for k, v in positions_dict.items() if k != target_sat_id}
        
        if len(other_sats) == 0:
            return risks
        
        # Process in chunks to manage memory
        chunk_size = 1000
        sat_ids = list(other_sats.keys())
        
        for i in range(0, len(sat_ids), chunk_size):
            chunk_ids = sat_ids[i:i+chunk_size]
            chunk_positions = np.array([other_sats[sat_id] for sat_id in chunk_ids])
            
            if self.gpu_available:
                try:
                    # Use GPU for distance computation
                    with tf.device('/GPU:0'):
                        distances_tensor = self.gpu_distance_matrix(target_positions, chunk_positions)
                        distances = distances_tensor.numpy()
                except:
                    # Fallback to CPU if GPU fails
                    distances = self._cpu_distance_computation(target_positions, chunk_positions)
            else:
                distances = self._cpu_distance_computation(target_positions, chunk_positions)
            
            # Find close approaches
            for j, sat_id in enumerate(chunk_ids):
                min_distance = np.min(distances[j])
                if min_distance < threshold_km:
                    min_time_idx = np.argmin(distances[j])
                    risks.append({
                        'debris_id': sat_id,
                        'min_distance_km': float(min_distance),
                        'time_index': int(min_time_idx),
                        'all_distances': distances[j].tolist()
                    })
        
        return risks
    
    def _cpu_distance_computation(self, target_positions, debris_positions):
        """CPU fallback for distance computation"""
        distances = []
        for debris_pos in debris_positions:
            diff = target_positions - debris_pos
            dist = np.linalg.norm(diff, axis=1)
            distances.append(dist)
        return np.array(distances)
    
    def analyze_collision_risk_fast(self, sat_id, debris_df, max_objects=5000):
        """
        Fast collision risk analysis with GPU acceleration
        Limited to fewer objects for faster processing
        """
        print(f"Starting fast collision risk analysis for satellite {sat_id}")
        start_time = time.time()
        
        # Limit dataset size for faster processing
        if len(debris_df) > max_objects:
            print(f"Limiting analysis to {max_objects} objects for faster processing")
            debris_df = debris_df.head(max_objects)
        
        # Create satellite objects
        satellite_objects = {}
        
        for _, row in debris_df.iterrows():
            try:
                sat_obj = EarthSatellite(row['tle_line1'], row['tle_line2'], row['name'])
                # Store with both original and normalized IDs
                norad_id = str(row['norad_id'])
                satellite_objects[norad_id] = sat_obj
                # Also store normalized version (remove leading zeros)
                normalized_id = str(int(norad_id))
                if normalized_id != norad_id:
                    satellite_objects[normalized_id] = sat_obj
            except Exception as e:
                continue
        
        # Try to find target satellite with different formatting
        target_found = False
        original_sat_id = str(sat_id)
        
        # Try multiple possible formats
        possible_formats = [
            original_sat_id,                    # Original format
            original_sat_id.zfill(5),          # Zero-padded to 5 digits
            str(int(original_sat_id)),         # Remove any leading zeros
            original_sat_id.lstrip('0')        # Remove leading zeros
        ]
        
        for possible_id in possible_formats:
            if possible_id in satellite_objects:
                sat_id = possible_id  # Use the format that exists
                target_found = True
                print(f"Found target satellite using ID format: {possible_id}")
                break
        
        if not target_found:
            print(f"Warning: Target satellite {original_sat_id} not found in satellite_objects")
            print(f"Tried formats: {possible_formats}")
            print(f"Available satellites (first 10): {list(satellite_objects.keys())[:10]}")
            # Check if any satellite IDs contain the target ID
            matches = [sid for sid in satellite_objects.keys() if original_sat_id in sid]
            if matches:
                print(f"Possible matches found: {matches[:5]}")
        
        print(f"Created {len(satellite_objects)} satellite objects")
        
        # Compute positions (reduced time range for speed)
        positions_dict, times = self.batch_position_computation(satellite_objects, hours=24, step=4)
        
        position_time = time.time()
        print(f"Position computation completed in {position_time - start_time:.2f} seconds")
        
        # Find close approaches
        risks = self.find_close_approaches_optimized(str(sat_id), positions_dict)
        
        analysis_time = time.time()
        print(f"Risk analysis completed in {analysis_time - position_time:.2f} seconds")
        print(f"Total analysis time: {analysis_time - start_time:.2f} seconds")
        print(f"Found {len(risks)} potential collision risks")
        
        # Convert to DataFrame and sort by minimum distance
        if risks:
            risks_df = pd.DataFrame(risks)
            risks_df = risks_df.sort_values('min_distance_km').head(20)  # Top 20 risks
            
            # Add satellite names
            name_mapping = dict(zip(debris_df['norad_id'].astype(str), debris_df['name']))
            risks_df['debris_name'] = risks_df['debris_id'].map(name_mapping)
            
            return risks_df
        else:
            return pd.DataFrame()
    
    def create_risk_summary(self, risks_df):
        """Create a summarized risk table"""
        if risks_df.empty:
            return "No collision risks detected within the specified threshold."
        
        # Create summary with key information
        summary = risks_df[['debris_name', 'debris_id', 'min_distance_km']].copy()
        summary['risk_level'] = summary['min_distance_km'].apply(
            lambda x: 'HIGH' if x < 10 else 'MEDIUM' if x < 25 else 'LOW'
        )
        
        return summary.head(10)

def quick_risk_analysis(sat_id, max_objects=3000):
    """
    Quick entry point for risk analysis
    Optimized for speed over completeness
    """
    try:
        # Load data
        debris_df = pd.read_csv('data/orbital_params.csv', dtype={'norad_id': str})
        
        # Initialize analyzer
        analyzer = GPUAcceleratedRiskAnalysis()
        
        # Run analysis
        risks_df = analyzer.analyze_collision_risk_fast(sat_id, debris_df, max_objects)
        summary = analyzer.create_risk_summary(risks_df)
        
        return {
            'risks_df': risks_df,
            'summary': summary,
            'total_risks': len(risks_df)
        }
        
    except Exception as e:
        print(f"Error in quick risk analysis: {e}")
        return {
            'risks_df': pd.DataFrame(),
            'summary': f"Error: {str(e)}",
            'total_risks': 0
        }

if __name__ == "__main__":
    # Test the optimized risk analysis
    result = quick_risk_analysis('25544')  # ISS
    print(f"Analysis complete. Found {result['total_risks']} risks.")
    if result['total_risks'] > 0:
        print("\nTop risks:")
        print(result['summary'])
