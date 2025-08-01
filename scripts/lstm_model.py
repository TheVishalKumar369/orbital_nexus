import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler
from skyfield.api import load
import pickle
import os
from skyfield.api import EarthSatellite

# Configure GPU settings
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print(f"GPU(s) available: {len(physical_devices)}")
    for gpu in physical_devices:
        print(f"  - {gpu}")
        # Enable memory growth to avoid allocating all GPU memory at once
        tf.config.experimental.set_memory_growth(gpu, True)
    # Set GPU as default device
    with tf.device('/GPU:0'):
        print("TensorFlow configured to use GPU:0")
else:
    print("No GPU available, using CPU")

class TrajectoryPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.ts = load.timescale()
        
    def compute_future_positions(self, sat_obj, hours=24, step=1):
        """Compute future positions for a satellite object with error handling"""
        try:
            times = self.ts.utc(*self.ts.now().utc[:4], range(0, hours, step))
            
            positions = []
            for t in times:
                try:
                    geocentric = sat_obj.at(t)
                    pos = geocentric.position.km
                    
                    # Validate position values
                    if np.any(np.isnan(pos)) or np.any(np.isinf(pos)):
                        # Use Earth's center as fallback position
                        pos = np.array([0.0, 0.0, 6371.0])  # Earth radius in km
                    
                    # Ensure reasonable bounds (within ~50,000 km of Earth)
                    pos_magnitude = np.linalg.norm(pos)
                    if pos_magnitude > 50000 or pos_magnitude < 100:
                        # Normalize to reasonable LEO altitude
                        pos = pos / pos_magnitude * 7000  # 700 km altitude
                    
                    positions.append(pos)
                    
                except Exception as e:
                    print(f"Error computing position at time {t}: {e}")
                    # Use default LEO position
                    positions.append(np.array([0.0, 0.0, 7000.0]))
            
            return np.array(positions, dtype=np.float64)
            
        except Exception as e:
            print(f"Error in compute_future_positions: {e}")
            # Return default positions for LEO orbit
            num_positions = max(1, hours // step)
            return np.array([[0.0, 0.0, 7000.0]] * num_positions, dtype=np.float64)
    
    def add_sat_obj_column(self, df):
        """Add satellite objects column from TLE lines"""
        print("Creating satellite objects from TLE lines...")
        
        # Check if required columns exist
        required_columns = ['tle_line1', 'tle_line2', 'name']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            print(f"Available columns: {list(df.columns)}")
            return df
        
        df['sat_obj'] = df.apply(
            lambda row: EarthSatellite(row['tle_line1'], row['tle_line2'], row['name']),
            axis=1
        )
        print(f"Successfully created {len(df)} satellite objects")
        return df
    
    def create_sequences(self, debris_df, norad_id, hours=24):
        """Create time-series sequences for LSTM training"""
        try:
            sat_row = debris_df[debris_df.norad_id == norad_id].iloc[0]
            
            # Check if sat_obj exists
            if 'sat_obj' not in sat_row:
                print(f"Missing sat_obj for {norad_id}")
                return None
                
            sat_obj = sat_row['sat_obj']
            
            # Get position history
            positions = self.compute_future_positions(sat_obj, hours=hours)
            
            # Get orbital features - check if they exist and validate
            feature_columns = ['semi_major_axis', 'eccentricity', 'inclination', 'mean_motion']
            missing_features = [col for col in feature_columns if col not in sat_row]
            
            if missing_features:
                print(f"Missing orbital features for {norad_id}: {missing_features}")
                # Use default values
                features = np.array([0.06, 0.01, 1.0, 15.0], dtype=np.float64)  # Default values
            else:
                features = sat_row[feature_columns].values.astype(np.float64)
                
                # Validate and clean feature values
                if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                    print(f"Invalid orbital features for {norad_id}, using defaults")
                    features = np.array([0.06, 0.01, 1.0, 15.0], dtype=np.float64)
                
                # Ensure reasonable bounds
                features[0] = max(0.01, min(features[0], 1.0))      # semi_major_axis
                features[1] = max(0.0, min(features[1], 0.99))      # eccentricity
                features[2] = max(0.0, min(features[2], np.pi))     # inclination
                features[3] = max(1.0, min(features[3], 20.0))      # mean_motion
            
            # Combine positions and features
            sequences = []
            for i in range(len(positions) - 1):
                seq = np.concatenate([positions[i], features])
                sequences.append(seq)
            
            return np.array(sequences)
            
        except Exception as e:
            print(f"Error creating sequences for {norad_id}: {e}")
            return None
    
    def prepare_training_data(self, debris_df, sample_size=1000):
        """Prepare training data for LSTM model"""
        print("Preparing training data...")
        
        # First, ensure we have satellite objects
        if 'sat_obj' not in debris_df.columns:
            print("Adding satellite objects from TLE lines...")
            debris_df = self.add_sat_obj_column(debris_df)
        debris_df['norad_id'] = debris_df['norad_id'].astype(str)
        
        X_sequences = []
        y_targets = []
        
        # Sample objects for training
        available_ids = debris_df.norad_id.dropna()
        if len(available_ids) == 0:
            print("No valid NORAD IDs found!")
            return None, None
            
        sample_size = min(sample_size, len(available_ids))
        sample_ids = available_ids.sample(sample_size)
        
        print(f"Processing {len(sample_ids)} satellites for training...")
        
        for norad_id in sample_ids:
            try:
                # Create input sequence
                seq = self.create_sequences(debris_df, str(norad_id), hours=24)
                if seq is None or len(seq) == 0:
                    continue
                
                # Get target position (6 hours ahead)
                sat_row = debris_df[debris_df.norad_id == str(norad_id)].iloc[0]
                target_pos = self.compute_future_positions(sat_row['sat_obj'], hours=6)[-1]
                
                X_sequences.append(seq)
                y_targets.append(target_pos)
                
            except Exception as e:
                print(f"Error processing {norad_id}: {e}")
                continue
        
        if len(X_sequences) == 0:
            print("No valid sequences created!")
            return None, None
        
        X_train = np.array(X_sequences, dtype=np.float64)
        y_train = np.array(y_targets, dtype=np.float64)
        print("X_train dtype:", X_train.dtype)
        print("y_train dtype:", y_train.dtype)
        
        print(f"Training data shape before cleaning: X={X_train.shape}, y={y_train.shape}")
        # Flatten for easier filtering
        X_flat = X_train.reshape(X_train.shape[0], -1)
        y_flat = y_train.reshape(y_train.shape[0], -1)
        valid_mask = (
            ~np.isnan(X_flat).any(axis=1) &
            ~np.isinf(X_flat).any(axis=1) &
            ~np.isnan(y_flat).any(axis=1) &
            ~np.isinf(y_flat).any(axis=1)
        )
        removed = np.sum(~valid_mask)
        if removed > 0:
            print(f"Filtered out {removed} samples due to NaN/Inf in X or y.")
        X_train = X_train[valid_mask]
        y_train = y_train[valid_mask]
        print(f"Training data shape after cleaning: X={X_train.shape}, y={y_train.shape}")
        
        if len(X_train) == 0:
            print("No valid sequences created after cleaning!")
            return None, None
        
        return X_train, y_train
    
    def build_model(self, input_shape):
        """Build LSTM model for trajectory prediction"""
        # Explicitly build model on GPU if available
        with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
            model = Sequential([
                LSTM(128, input_shape=input_shape, return_sequences=True),
                Dropout(0.2),
                LSTM(64),
                Dense(32, activation='relu'),
                Dense(3)  # Output: x,y,z coordinates
            ])
            
            model.compile(loss='mse', optimizer='adam', metrics=['mae'])
            print(f"Model built on device: {'/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'}")
            return model
    
    def train_model(self, debris_df, epochs=50, batch_size=32):
        """Train the LSTM model"""
        from app import add_log
        add_log("Training LSTM model...", level="info")
        # Prepare training data
        X_train, y_train = self.prepare_training_data(debris_df)
        if X_train is None:
            add_log("No training data available!", level="error")
            return False
        add_log(f"Training data shape after cleaning: X={X_train.shape}, y={y_train.shape}", level="info")
        # Scale features
        X_scaled = self.scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1]))
        X_scaled = X_scaled.reshape(X_train.shape)
        add_log("Data scaling complete.", level="info")
        # Data validation checks
        print("NaN in X_scaled:", np.isnan(X_scaled).any())
        print("Inf in X_scaled:", np.isinf(X_scaled).any())
        print("NaN in y_train:", np.isnan(y_train).any())
        print("Inf in y_train:", np.isinf(y_train).any())
        print("X_scaled min/max:", X_scaled.min(), X_scaled.max())
        print("y_train min/max:", y_train.min(), y_train.max())
        print("X_scaled shape:", X_scaled.shape)
        print("y_train shape:", y_train.shape)
        assert not np.isnan(X_scaled).any(), "NaN in X_scaled"
        assert not np.isnan(y_train).any(), "NaN in y_train"
        assert not np.isinf(X_scaled).any(), "Inf in X_scaled"
        assert not np.isinf(y_train).any(), "Inf in y_train"
        assert X_scaled.shape[0] == y_train.shape[0], "Mismatched samples between X and y"

        # Build and train model
        self.model = self.build_model((X_scaled.shape[1], X_scaled.shape[2]))
        add_log("Model architecture built. Starting training...", level="info")
        history = self.model.fit(
            X_scaled, 
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1
        )
        add_log("Model training complete. Saving model...", level="success")
        # Save model and scaler
        os.makedirs('models', exist_ok=True)
        self.model.save('models/debris_lstm.keras')
        with open('models/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        add_log("Model saved to models/debris_lstm.keras", level="success")
        
        # Create deployment info
        deployment_info = {
            'trained_on': 'GPU' if tf.config.list_physical_devices('GPU') else 'CPU',
            'model_path': 'models/debris_lstm.keras',
            'scaler_path': 'models/scaler.pkl',
            'training_samples': X_train.shape[0],
            'input_shape': X_train.shape[1:],
            'trained_at': str(pd.Timestamp.now())
        }
        
        with open('models/deployment_info.json', 'w') as f:
            import json
            json.dump(deployment_info, f, indent=2)
        
        add_log(f"Model ready for deployment (trained on {deployment_info['trained_on']})", level="success")
        return True
    
    def predict_trajectory(self, sat_obj, hours=24):
        """Predict trajectory for a satellite object"""
        if self.model is None:
            print("Model not trained! Please train the model first.")
            return None
        
        # Get current position and features
        current_pos = self.compute_future_positions(sat_obj, hours=1)[0]
        
        # For now, use dummy features (in practice, extract from sat_obj)
        features = np.array([1.0, 0.1, 0.5, 15.0])  # Example values
        
        # Create input sequence
        input_seq = np.concatenate([current_pos, features]).reshape(1, -1)
        input_scaled = self.scaler.transform(input_seq)
        
        # Predict
        prediction = self.model.predict(input_scaled)
        return prediction[0]

if __name__ == "__main__":
    # Load orbital parameters
    try:
        debris_df = pd.read_csv('data/orbital_params.csv')
        print(f"Loaded {len(debris_df)} satellite objects")
        print(f"Columns: {list(debris_df.columns)}")
        
        # Check for required columns
        required_columns = ['tle_line1', 'tle_line2', 'name']
        missing_columns = [col for col in required_columns if col not in debris_df.columns]
        
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            print("Please ensure TLE data has been parsed correctly.")
            exit(1)
            
    except FileNotFoundError:
        print("Orbital parameters file not found. Please run tle_parser.py first.")
        exit(1)
    
    # Initialize and train model
    predictor = TrajectoryPredictor()
    success = predictor.train_model(debris_df, epochs=10)  # Reduced epochs for testing
    
    if success:
        print("Model training completed successfully!")
    else:
        print("Model training failed!") 