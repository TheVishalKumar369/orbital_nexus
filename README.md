# 🚀 Space Debris Trajectory Prediction

A machine learning system for predicting future positions of space debris and assessing collision risk using Space-Track.org TLE data.

## 🌟 Features

- **Automated TLE Pipeline**: Daily updated debris catalog from Space-Track.org
- **LSTM Trajectory Forecast**: 48-hour position predictions using deep learning
- **Collision Risk Assessment**: 3D visualization of high-risk zones
- **Real-time Dashboard**: Interactive web interface for monitoring
- **Podman Containerization**: Easy deployment and scaling with rootless containers
- **FastAPI Backend**: Modern, high-performance API with automatic documentation

## 🛠️ Technology Stack

- **Python 3.9+**
- **FastAPI**: Modern, fast web framework with automatic API docs
- **Uvicorn**: ASGI server for FastAPI
- **TensorFlow/Keras**: LSTM neural networks
- **Skyfield**: Astronomical calculations
- **Space-Track API**: TLE data acquisition
- **Plotly**: Interactive 3D visualizations
- **Podman**: Rootless containerization

## 📋 Prerequisites

1. **Space-Track.org Account**: Register at [Space-Track.org](https://www.space-track.org/)
2. **Podman**: Install Podman and podman-compose
3. **Git**: Clone this repository

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone <repository-url>
cd space-debris-tracker
```

### 2. Configure Environment
```bash
# Copy environment template
cp env.example .env

# Edit .env with your Space-Track credentials
nano .env
```

Add your Space-Track.org credentials:
```
SPACETRACK_USERNAME=your_username_here
SPACETRACK_PASSWORD=your_password_here
FLASK_SECRET_KEY=your-secret-key-here
```

### 3. Build and Run with Podman
```bash
# Build the Podman image
podman-compose build

# Start the application (GPU support)
podman-compose up -d

# Or start with CPU-only mode
podman-compose -f podman-compose.cpu.yml up -d

# View logs
podman-compose logs -f
```

### 4. Access the Application
Open your browser and navigate to: `http://localhost:5000`

## 📊 Usage

### Web Dashboard
1. **Download TLE Data**: Fetch latest debris data from Space-Track.org
2. **Parse Data**: Extract orbital parameters from TLE data
3. **Train Model**: Train LSTM model for trajectory prediction
4. **Analyze Risks**: Generate collision risk assessment

### API Endpoints
- `GET /`: Main dashboard page
- `GET /docs`: Interactive API documentation (Swagger UI)
- `GET /health`: Health check endpoint
- `GET /api/status`: System status and data availability
- `POST /api/download-data`: Download TLE data
- `POST /api/parse-data`: Parse TLE data
- `POST /api/train-model`: Train ML model
- `POST /api/analyze-risks`: Analyze collision risks
- `GET /api/satellites`: List available satellites

### API Documentation
FastAPI automatically generates interactive API documentation. Visit `/docs` for the Swagger UI or `/redoc` for ReDoc documentation.

## 🏗️ Project Structure

```
space-debris-tracker/
├── data/                   # Data storage
│   ├── debris_tle.txt     # Raw TLE data
│   └── orbital_params.csv # Processed orbital parameters
├── models/                 # ML models
│   ├── debris_lstm.h5     # Trained LSTM model
│   └── scaler.pkl         # Data scaler
├── scripts/               # Core processing scripts
│   ├── space_track.py     # TLE data acquisition
│   ├── tle_parser.py      # TLE data parsing
│   ├── lstm_model.py      # ML model training
│   └── dashboard.py       # Risk assessment
├── templates/             # Web templates
│   └── index.html         # Main dashboard
├── app.py                 # FastAPI application
├── requirements.txt       # Python dependencies
├── Containerfile         # Podman container configuration
├── podman-compose.yml    # Podman Compose setup
├── env.example           # Environment template
└── README.md             # This file
```

## 🔧 Development

### Local Development Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

### Running Individual Scripts
```bash
# Download TLE data
python scripts/space_track.py

# Parse TLE data
python scripts/tle_parser.py

# Train ML model
python scripts/lstm_model.py

# Generate risk dashboard
python scripts/dashboard.py
```

### FastAPI Development
```bash
# Run with auto-reload for development
uvicorn app:app --reload --host 0.0.0.0 --port 5000

# Run with production settings
uvicorn app:app --host 0.0.0.0 --port 5000 --workers 4
```

## 📈 Advanced Features

### Live ISS Monitoring
```python
from skyfield.api import load
stations_url = 'http://celestrak.org/NORAD/elements/stations.txt'
satellites = load.tle_file(stations_url)
iss = [sat for sat in satellites if sat.name == 'ISS (ZARYA)'][0]
```

### Monte Carlo Risk Simulation
- Propagate orbital uncertainty using covariance matrices
- Generate probability distributions for collision risks

### NASA CDM Integration
- Use Conjunction Data Messages for validation
- Compare predictions with official risk assessments

## ⚠️ Important Notes

### Space-Track Rate Limits
- Maximum 100 requests/minute
- Maximum 2000 requests/day
- Cache data locally to avoid rate limiting

### Computational Considerations
- Use GPU acceleration for large datasets
- Implement parallel processing for debris analysis
- Consider cloud deployment for scalability

### Physical Models
- Augment ML with SGP4 orbital propagator
- Ensure physics consistency in predictions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Space-Track.org](https://www.space-track.org/) for TLE data
- [Skyfield](https://rhodesmill.org/skyfield/) for astronomical calculations
- [NASA](https://www.nasa.gov/) for orbital mechanics research

## 📞 Support

For questions or issues:
1. Check the [Issues](https://github.com/your-repo/issues) page
2. Create a new issue with detailed description
3. Contact the development team

---

**Note**: This system is for educational and research purposes. For operational space debris tracking, consult official sources like NASA's Orbital Debris Program Office. 