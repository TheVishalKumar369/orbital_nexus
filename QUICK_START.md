# 🚀 Quick Start Guide

## Prerequisites
1. **Podman & Podman Compose** installed
2. **Space-Track.org account** (free registration)

## Setup Steps

### 1. Configure Environment
```bash
# Copy environment template
cp env.example .env

# Edit .env with your Space-Track credentials
# Replace the placeholder values with your actual credentials
```

### 2. Run the Startup Script
```bash
python start.py
```

This script will:
- ✅ Check Podman installation
- ✅ Verify .env configuration
- ✅ Create necessary directories
- ✅ Build and start the Podman container
- ✅ Provide access instructions

### 3. Access the Application
Open your browser and go to: **http://localhost:5000**

## Using the Web Dashboard

### Step 1: Download TLE Data
- Click "Download TLE Data" button
- This fetches the latest debris catalog from Space-Track.org
- Wait for the success message

### Step 2: Parse Data
- Click "Parse Data" button
- This processes the TLE data and extracts orbital parameters
- Shows the number of objects processed

### Step 3: Train Model
- Click "Train Model" button
- This trains the LSTM neural network for trajectory prediction
- May take several minutes depending on data size

### Step 4: Analyze Risks
- Click "Analyze Risks" button
- This generates collision risk assessment for ISS (25544)
- Shows 3D visualization and risk table

## API Documentation

FastAPI provides automatic API documentation:

- **Swagger UI**: Visit `http://localhost:5000/docs`
- **ReDoc**: Visit `http://localhost:5000/redoc`
- **Health Check**: Visit `http://localhost:5000/health`

## Podman Commands

```bash
# Start the application
podman-compose up -d

# View logs
podman-compose logs -f

# Stop the application
podman-compose down

# Rebuild and restart
podman-compose up --build -d
```

## FastAPI Development

```bash
# Run with auto-reload (development)
uvicorn app:app --reload --host 0.0.0.0 --port 5000

# Run with production settings
uvicorn app:app --host 0.0.0.0 --port 5000 --workers 4
```

## Troubleshooting

### Common Issues

1. **Podman not running**
   - Start Podman service
   - Ensure Podman is properly configured

2. **Space-Track API errors**
   - Verify credentials in .env file
   - Check Space-Track.org account status
   - Respect rate limits (100 req/min, 2000 req/day)

3. **Port 5000 already in use**
   - Change port in podman-compose.yml
   - Or stop other services using port 5000

4. **Memory issues during training**
   - Reduce sample size in lstm_model.py
   - Use smaller epochs for testing

5. **FastAPI/Uvicorn issues**
   - Check Python version (3.9+ required)
   - Verify all dependencies are installed
   - Check logs for specific error messages

### Getting Help

- Check the system log in the web dashboard
- View Podman logs: `podman-compose logs -f`
- Access API documentation at `/docs`
- Review the main README.md for detailed documentation

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Main dashboard |
| GET | `/docs` | API documentation (Swagger) |
| GET | `/health` | Health check |
| GET | `/api/status` | System status |
| POST | `/api/download-data` | Download TLE data |
| POST | `/api/parse-data` | Parse TLE data |
| POST | `/api/train-model` | Train ML model |
| POST | `/api/analyze-risks` | Analyze collision risks |
| GET | `/api/satellites` | List satellites |

## Next Steps

After successful setup:
1. Explore the web dashboard features
2. Check out the API documentation at `/docs`
3. Try different satellite IDs for risk analysis
4. Experiment with model parameters
5. Consider production deployment options

---

**Note**: This is a research/educational system. For operational space debris tracking, consult official sources. 