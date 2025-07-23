🚀 Project Roadmap: Space Debris Trajectory Prediction
Goal: Predict future positions of space debris and assess collision risk using ML
Datasets: Space-Track.org TLE Data
Toolkit: Python + skyfield, spacetrack, scikit-learn, tensorflow, and plotly
Output: Interactive collision risk dashboard and trajectory forecast model

1. Data Acquisition from Space-Track.org
Step 1: Install Required Libraries
bash
pip install spacetrack skyfield numpy pandas scikit-learn tensorflow plotly
Step 2: API Authentication
Create space_track.py:

python
from spacetrack import SpaceTrackClient

st = SpaceTrackClient(identity='your_username', password='your_password')


#inside the .env file 
your_username  = "panchanarayansahu00@gmail.com"
your_password = "password_TRACKER"

# Download Two-Line Element (TLE) data for all debris
tle_data = st.tle(
    iter_lines=True, 
    norad_cat_id=range(1, 50000),  # All objects
    epoch='>now-30',                # Last 30 days
    format='tle'
)

# Save to file
with open('debris_tle.txt', 'w') as f:
    for line in tle_data:
        f.write(line + '\n')
2. Data Preprocessing
Step 1: Parse TLE Data
Create tle_parser.py:

python
import pandas as pd
from skyfield.api import EarthSatellite, load

debris_list = []
with open('debris_tle.txt') as f:
    lines = f.readlines()

# Parse every 3 lines (name + line1 + line2)
for i in range(0, len(lines), 3):
    name = lines[i].strip()
    line1 = lines[i+1].strip()
    line2 = lines[i+2].strip()
    
    satellite = EarthSatellite(line1, line2, name)
    debris_list.append({
        'norad_id': line2[2:7],
        'name': name,
        'tle_line1': line1,
        'tle_line2': line2,
        'sat_obj': satellite  # Skyfield object
    })

debris_df = pd.DataFrame(debris_list)
Step 2: Extract Orbital Parameters
python
from skyfield.api import load

ts = load.timescale()

def get_orbital_elements(row):
    sat = row['sat_obj']
    t = ts.now()  # Current time
    geocentric = sat.at(t)
    
    # Extract orbital elements
    position = geocentric.position.km
    velocity = geocentric.velocity.km_per_s
    
    return pd.Series({
        'semi_major_axis': sat.model.no_kozai,  # Semi-major axis (Earth radii)
        'eccentricity': sat.model.ecco,
        'inclination': sat.model.inclo,
        'raan': sat.model.nodeo,  # Right Ascension of Ascending Node
        'arg_perigee': sat.model.argpo,
        'mean_motion': sat.model.no  # Revolutions/day
    })

debris_df = debris_df.join(debris_df.apply(get_orbital_elements, axis=1))
3. Feature Engineering
Step 1: Compute Future Positions
python
import numpy as np

def compute_future_positions(norad_id, hours=24, step=1):
    sat = debris_df[debris_df.norad_id==norad_id].iloc[0]['sat_obj']
    times = ts.utc(*ts.now().utc[:4], range(0, hours, step))
    
    positions = []
    for t in times:
        geocentric = sat.at(t)
        positions.append(geocentric.position.km)
    
    return np.array(positions)

# Example: 24-hour trajectory for object 25544 (ISS)
iss_trajectory = compute_future_positions('25544')
Step 2: Create Training Dataset
python
from sklearn.preprocessing import StandardScaler

# Target variable: Position at t+6 hours
X = debris_df[['semi_major_axis', 'eccentricity', 'inclination', 'mean_motion']]
y = debris_df.apply(lambda row: compute_future_positions(row['norad_id'], hours=6)[-1], axis=1)

# Convert y to 3 columns (x,y,z)
y = np.array(y.tolist())

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
4. ML Model: LSTM for Trajectory Prediction
Step 1: Build Sequence Model
Create lstm_model.py:

python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

model = Sequential([
    LSTM(128, input_shape=(24, 4), return_sequences=True),  # 24 time steps, 4 features
    Dropout(0.2),
    LSTM(64),
    Dense(32, activation='relu'),
    Dense(3)  # Output: x,y,z coordinates
])

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
Step 2: Create Time-Series Sequences
python
# For each object, create 24-hour position history
def create_sequences(norad_id):
    positions = compute_future_positions(norad_id, hours=24)
    features = debris_df[debris_df.norad_id==norad_id][['semi_major_axis', 'eccentricity', 'inclination', 'mean_motion']].values
    return np.hstack([positions[:-1], np.tile(features, (len(positions)-1, 1))])

# Build full dataset
X_sequences = []
y_targets = []

for norad_id in debris_df.norad_id.sample(1000):  # Use subset for training
    seq = create_sequences(norad_id)
    target = compute_future_positions(norad_id, hours=6)[-1]  # Position at 30 hours
    X_sequences.append(seq)
    y_targets.append(target)
    
X_train = np.array(X_sequences)
y_train = np.array(y_targets)
Step 3: Train Model
python
history = model.fit(
    X_train, 
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2
)
5. Collision Risk Assessment
Step 1: Compute Close Approaches
python
from scipy.spatial.distance import cdist

def find_conjunctions(sat_id, threshold_km=100):
    sat_positions = compute_future_positions(sat_id, hours=48)
    risks = []
    
    for time_idx, sat_pos in enumerate(sat_positions):
        for _, debris in debris_df.iterrows():
            if debris['norad_id'] == sat_id: 
                continue
                
            debris_pos = compute_future_positions(debris['norad_id'], hours=48)[time_idx]
            distance = np.linalg.norm(sat_pos - debris_pos)
            
            if distance < threshold_km:
                risks.append({
                    'time_index': time_idx,
                    'debris_id': debris['norad_id'],
                    'distance_km': distance,
                    'relative_velocity': np.linalg.norm(
                        sat.sat_obj.at(ts.utc(*ts.now().utc[:4], time_idx)).velocity.km_per_s - 
                        debris['sat_obj'].at(ts.utc(*ts.now().utc[:4], time_idx)).velocity.km_per_s
                    )
                })
    return pd.DataFrame(risks)

# Example: Check collisions for ISS (25544)
iss_risks = find_conjunctions('25544')
Step 2: Build Risk Dashboard
Create dashboard.py:

python
import plotly.express as px
import plotly.graph_objects as go

# 3D Trajectory Plot
fig = go.Figure()
fig.add_trace(go.Scatter3d(
    x=iss_trajectory[:,0], y=iss_trajectory[:,1], z=iss_trajectory[:,2],
    mode='lines', name='ISS Trajectory'
))

# Add collision points
risk_points = iss_risks.merge(debris_df, left_on='debris_id', right_on='norad_id')
fig.add_trace(go.Scatter3d(
    x=risk_points['position_x'],
    y=risk_points['position_y'],
    z=risk_points['position_z'],
    mode='markers',
    marker=dict(size=risk_points['distance_km']/10, color='red'),
    name='Collision Risk'
))

fig.update_layout(title='ISS Collision Risk Assessment (48 hours)')
fig.show()

# Risk Table
risk_table = risk_points[['name', 'distance_km', 'relative_velocity']].sort_values('distance_km')
print(risk_table.head(10))
6. Deployment & Portfolio Presentation
GitHub Repository Structure:
text
space-debris-tracker/
├── data/
│   ├── debris_tle.txt
│   └── orbital_params.csv
├── notebooks/
│   ├── Data_Acquisition.ipynb
│   ├── Trajectory_Prediction.ipynb
│   └── Collision_Risk.ipynb
├── scripts/
│   ├── tle_parser.py
│   ├── lstm_model.py
│   └── dashboard.py
├── models/
│   └── debris_lstm.h5
└── README.md
Key Features to Highlight:
Automated TLE Pipeline: Daily updated debris catalog

LSTM Trajectory Forecast: 48-hour position predictions

Collision Heatmaps: 3D visualization of high-risk zones

Risk Prioritization: Sort by distance/relative velocity

🔭 Advanced Extensions
Live ISS Risk Monitoring:

python
# Real-time ISS position (every 5 min)
from skyfield.api import load
stations_url = 'http://celestrak.org/NORAD/elements/stations.txt'
satellites = load.tle_file(stations_url)
iss = [sat for sat in satellites if sat.name == 'ISS (ZARYA)'][0]
NASA CDM Integration:
Use Conjunction Data Messages for validation

Monte Carlo Risk Simulation:
Propagate orbital uncertainty using covariance matrices

🚨 Critical Notes
Space-Track Rate Limits:

Max 100 requests/minute, 2000/day

Cache data locally to avoid bans

Computational Optimization:
Use GPU acceleration and parallel processing for large debris sets

Physical Models:
Augment ML with SGP4 orbital propagator for physics consistency