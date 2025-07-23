from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import os
import pandas as pd
import numpy as np
from scripts.space_track import download_tle_data
from scripts.tle_parser import parse_tle_data, extract_orbital_elements
from scripts.lstm_model import TrajectoryPredictor
from scripts.dashboard import CollisionRiskDashboard
from dotenv import load_dotenv
import uvicorn
from enhanced_routes import router as enhanced_router
from fastapi_limiter import FastAPILimiter
try:
    from fastapi_limiter.depends import RateLimiter, get_remote_address
except ImportError:
    from fastapi_limiter.depends import RateLimiter
    # Fallback: define get_remote_address if not present
    async def get_remote_address(request):
        return request.client.host
import redis.asyncio as aioredis
from starlette.requests import Request as StarletteRequest

import asyncio
import threading
from datetime import datetime
import json
import time

# Load environment variables
load_dotenv()

# GPU Configuration (will be initialized after log functions are defined)
GPU_CONFIGURED = False

# Global status management functions
def load_global_status():
    """Load global status from JSON file"""
    try:
        with open('data/global_status.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        # Create default status if file doesn't exist
        default_status = {
            "last_tle_download": 0,
            "last_model_training": 0,
            "tle_cooldown_seconds": 5400,
            "training_cooldown_seconds": 3600,
            "download_count": 0,
            "training_count": 0
        }
        save_global_status(default_status)
        return default_status

def save_global_status(status):
    """Save global status to JSON file"""
    os.makedirs('data', exist_ok=True)
    with open('data/global_status.json', 'w') as f:
        json.dump(status, f, indent=2)

def update_global_status(key, value):
    """Update a specific key in global status"""
    status = load_global_status()
    status[key] = value
    save_global_status(status)

def get_time_until_next_download():
    """Get time remaining until next TLE download is allowed"""
    status = load_global_status()
    now = int(time.time())
    time_since_download = now - status["last_tle_download"]
    cooldown = status["tle_cooldown_seconds"]
    remaining = max(0, cooldown - time_since_download)
    return remaining

def get_time_until_next_training():
    """Get time remaining until next model training is allowed"""
    status = load_global_status()
    now = int(time.time())
    time_since_training = now - status["last_model_training"]
    cooldown = status["training_cooldown_seconds"]
    remaining = max(0, cooldown - time_since_training)
    return remaining

def format_time_remaining(seconds):
    """Format seconds into human-readable time"""
    if seconds == 0:
        return "Ready now"
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"

# Initialize FastAPI app
app = FastAPI(
    title="Space Debris Tracker API",
    description="A machine learning system for predicting space debris trajectories and collision risks",
    version="1.0.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Pydantic models for request/response
class SatelliteRequest(BaseModel):
    sat_id: str = "25544"  # Default to ISS

class RiskAnalysisResponse(BaseModel):
    status: str
    total_risks: int
    risk_table: list

class StatusResponse(BaseModel):
    tle_data: bool
    orbital_params: bool
    model: bool
    scaler: bool
    object_count: int

# Simple in-memory log buffer (thread-safe)
log_buffer = []
log_lock = threading.Lock()
LOG_BUFFER_SIZE = 200  # Keep last 200 log messages
LOG_FILE = 'data/global_logs.json'

def load_logs_from_disk():
    """Load logs from disk into the in-memory buffer on startup."""
    global log_buffer
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as f:
            try:
                log_buffer = json.load(f)
            except Exception:
                log_buffer = []
    else:
        log_buffer = []

def save_logs_to_disk():
    """Save the current log buffer to disk."""
    with open(LOG_FILE, 'w') as f:
        json.dump(log_buffer, f, indent=2)

def add_log(message, level="info"):
    global log_buffer
    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "message": message,
        "level": level
    }
    with log_lock:
        log_buffer.append(entry)
        if len(log_buffer) > LOG_BUFFER_SIZE:
            log_buffer = log_buffer[-LOG_BUFFER_SIZE:]
        save_logs_to_disk()

# Load logs from disk on startup
load_logs_from_disk()

# GPU Configuration Initialization
try:
    from gpu_config import configure_tensorflow_for_production
    GPU_CONFIGURED = configure_tensorflow_for_production()
    add_log("GPU configuration loaded successfully", level="info")
except Exception as gpu_error:
    # Continue without GPU optimization if configuration fails
    print(f"Warning: GPU configuration failed: {gpu_error}")
    import tensorflow as tf
    # Basic TensorFlow setup without GPU optimization
    tf.get_logger().setLevel('INFO')

# Include enhanced routes
app.include_router(enhanced_router)
@app.on_event("startup")
async def startup():
    redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
    redis = await aioredis.from_url(redis_url, encoding="utf8", decode_responses=True)
    await FastAPILimiter.init(redis)

# Custom rate limiter for download endpoint
async def global_and_rate_limit(request: Request):
    cooldown = get_time_until_next_download()
    if cooldown > 0:
        # Skip rate limiting, let endpoint handle cooldown in the route
        return
    # If cooldown is over, apply rate limiting
    limiter = RateLimiter(times=20, seconds=60, identifier=get_remote_address)
    return await limiter(request)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Cyberpunk-themed main dashboard page"""
    return templates.TemplateResponse("cyberpunk_index.html", {"request": request})

@app.get("/classic", response_class=HTMLResponse)
async def classic_index(request: Request):
    """Classic dashboard page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/download-data")
async def download_data(
    request: Request,
    rate_limiter=Depends(RateLimiter(times=20, seconds=60, identifier=get_remote_address))
):
    tle_file = 'data/debris_tle.txt'
    tle_data = ""
    cooldown = get_time_until_next_download()
    if os.path.exists(tle_file):
        with open(tle_file, 'r') as f:
            tle_data = f.read()
    if cooldown > 0:
        add_log(f"TLE data is up-to-date. Next update available in {format_time_remaining(cooldown)}.", level="info")
        return JSONResponse({
            "status": "cached",
            "message": f"TLE data is up-to-date. Next update available in {format_time_remaining(cooldown)}.",
            "cooldown_seconds": cooldown,
            "tle_data": tle_data
        })
    # If cooldown is over, rate limiting will be applied by Depends
    add_log("Downloading TLE data from Space-Track.org...", level="info")
    success = download_tle_data()
    if success:
        add_log("TLE data downloaded successfully", level="success")
        status = load_global_status()
        status["last_tle_download"] = int(time.time())
        status["download_count"] += 1
        save_global_status(status)
        if os.path.exists(tle_file):
            with open(tle_file, 'r') as f:
                tle_data = f.read()
        return JSONResponse({
            "status": "success",
            "message": "TLE data downloaded successfully",
            "cooldown_seconds": 5400,
            "tle_data": tle_data,
            "total_downloads": status["download_count"]
        })
    else:
        add_log("Failed to download TLE data. Serving cached data.", level="error")
        return JSONResponse({
            "status": "error",
            "message": "Failed to download TLE data. Serving cached data.",
            "cooldown_seconds": 5400,
            "tle_data": tle_data
        })

@app.exception_handler(429)
async def rate_limit_exceeded_handler(request: StarletteRequest, exc):
    return HTMLResponse(
        """
        Rate limit exceeded: Only 20 requests are allowed per minute. Since other people across world are using it so catch up later.<br><br>
        <i>With great power comes great responsibility</i>
        """,
        status_code=429
    )

@app.post("/api/parse-data")
async def parse_data():
    """Parse TLE data and extract orbital parameters"""
    try:
        add_log("Parsing TLE data and extracting orbital parameters...", level="info")
        debris_df = parse_tle_data()
        if debris_df is not None:
            debris_df = extract_orbital_elements(debris_df)
            add_log(f"Parsed {len(debris_df)} satellite objects", level="success")
            return JSONResponse({
                "status": "success", 
                "message": f"Parsed {len(debris_df)} satellite objects"
            })
        else:
            add_log("Failed to parse TLE data", level="error")
            return JSONResponse({
                "status": "error", 
                "message": "Failed to parse TLE data"
            })
    except Exception as e:
        add_log(f"Error parsing TLE data: {str(e)}", level="error")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/train-model")
async def train_model():
    """Train the LSTM model"""
    try:
        add_log("Training the LSTM model...", level="info")
        # Load orbital parameters
        debris_df = pd.read_csv('data/orbital_params.csv')
        
        # Check for required columns
        required_columns = ['tle_line1', 'tle_line2', 'name']
        missing_columns = [col for col in required_columns if col not in debris_df.columns]
        
        if missing_columns:
            error_msg = f"Missing required columns: {missing_columns}. Please ensure TLE data has been parsed correctly."
            add_log(error_msg, level="error")
            return JSONResponse({
                "status": "error", 
                "message": error_msg
            })
        
        # Initialize and train model
        predictor = TrajectoryPredictor()
        success = predictor.train_model(debris_df, epochs=10)  # Reduced for demo
        if success:
            add_log("Model trained successfully", level="success")
            # Update global status
            status = load_global_status()
            status["last_model_training"] = int(time.time())
            status["training_count"] += 1
            save_global_status(status)
            return JSONResponse({
                "status": "success", 
                "message": "Model trained successfully",
                "total_trainings": status["training_count"]
            })
        else:
            add_log("Failed to train model", level="error")
            return JSONResponse({
                "status": "error", 
                "message": "Failed to train model"
            })
    except Exception as e:
        add_log(f"Error training model: {str(e)}", level="error")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze-risks")
async def analyze_risks(request: SatelliteRequest):
    """Analyze collision risks for a satellite"""
    try:
        add_log(f"Analyzing collision risks for satellite {request.sat_id}...", level="info")
        dashboard = CollisionRiskDashboard()
        results = dashboard.generate_dashboard(str(request.sat_id))
        if results:
            # Convert risk table to JSON-serializable format
            risk_data = results['risk_table']
            # Ensure it's a list (it should already be from the updated dashboard)
            if not isinstance(risk_data, list):
                if isinstance(risk_data, pd.DataFrame):
                    risk_data = risk_data.to_dict('records')
                elif isinstance(risk_data, str):
                    risk_data = [{'message': risk_data}]
                else:
                    risk_data = []
            
            add_log(f"Collision risk analysis complete for satellite {request.sat_id}", level="success")
            return RiskAnalysisResponse(
                status="success",
                total_risks=results['total_risks'],
                risk_table=risk_data
            )
        else:
            add_log(f"Failed to analyze risks for satellite {request.sat_id}", level="error")
            return JSONResponse({
                "status": "error", 
                "message": "Failed to analyze risks"
            })
    except Exception as e:
        add_log(f"Error analyzing risks: {str(e)}", level="error")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/satellites")
async def get_satellites():
    """Get list of available satellites"""
    try:
        debris_df = pd.read_csv('data/orbital_params.csv', dtype={'norad_id': str})
        satellites = debris_df[['norad_id', 'name']].head(100).to_dict('records')
        return JSONResponse({
            "status": "success", 
            "satellites": satellites
        })
    except FileNotFoundError:
        return JSONResponse({
            "status": "error", 
            "message": "No satellite data available"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/global-status")
async def get_global_status():
    """Get global status including last download/training times and countdowns"""
    status = load_global_status()
    now = int(time.time())
    
    # Calculate time since last actions
    time_since_download = now - status["last_tle_download"]
    time_since_training = now - status["last_model_training"]
    
    # Format timestamps
    last_download_time = datetime.fromtimestamp(status["last_tle_download"]).strftime("%Y-%m-%d %H:%M:%S UTC") if status["last_tle_download"] > 0 else "Never"
    last_training_time = datetime.fromtimestamp(status["last_model_training"]).strftime("%Y-%m-%d %H:%M:%S UTC") if status["last_model_training"] > 0 else "Never"
    
    return JSONResponse({
        "last_tle_download": {
            "timestamp": status["last_tle_download"],
            "formatted": last_download_time,
            "time_since": time_since_download,
            "cooldown_remaining": get_time_until_next_download(),
            "cooldown_formatted": format_time_remaining(get_time_until_next_download()),
            "can_download": get_time_until_next_download() == 0
        },
        "last_model_training": {
            "timestamp": status["last_model_training"],
            "formatted": last_training_time,
            "time_since": time_since_training,
            "cooldown_remaining": get_time_until_next_training(),
            "cooldown_formatted": format_time_remaining(get_time_until_next_training()),
            "can_train": get_time_until_next_training() == 0
        },
        "counts": {
            "download_count": status["download_count"],
            "training_count": status["training_count"]
        }
    })

@app.get("/api/status", response_model=StatusResponse)
async def get_status():
    """Get system status"""
    status = {
        "tle_data": os.path.exists('data/debris_tle.txt'),
        "orbital_params": os.path.exists('data/orbital_params.csv'),
        "model": os.path.exists('models/debris_lstm.h5'),
        "scaler": os.path.exists('models/scaler.pkl'),
        "object_count": 0
    }
    
    # Count objects if available
    if status["orbital_params"]:
        try:
            debris_df = pd.read_csv('data/orbital_params.csv', dtype={'norad_id': str})
            status["object_count"] = len(debris_df)
        except:
            status["object_count"] = 0
    
    return StatusResponse(**status)

@app.get("/api/logs")
async def get_logs():
    """Return recent log messages for the frontend log panel"""
    with log_lock:
        return {"logs": list(log_buffer)}

@app.get("/api/gpu-status")
async def get_gpu_status():
    """Get GPU status and configuration"""
    from gpu_config import get_device_info, check_gpu_environment
    
    device_info = get_device_info()
    gpu_env = check_gpu_environment()
    
    return JSONResponse({
        "device_info": device_info,
        "environment_variables": gpu_env,
        "gpu_available": device_info['gpu_count'] > 0,
        "using_gpu": device_info['gpu_count'] > 0 and device_info['cuda_available']
    })

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "space-debris-tracker"}

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    # Run with uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=5000,
        reload=True,
        log_level="info"
    ) 