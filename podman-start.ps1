# PowerShell Podman startup script for Space Debris Tracking System

Write-Host "Starting Space Debris Tracking System with Podman..." -ForegroundColor Green

# Check if Podman is installed
try {
    podman --version | Out-Null
    Write-Host "Podman is installed" -ForegroundColor Green
} catch {
    Write-Host "Error: Podman is not installed. Please install Podman first." -ForegroundColor Red
    Write-Host "Download from: https://podman.io/getting-started/installation" -ForegroundColor Yellow
    exit 1
}

# Check if podman-compose is installed
try {
    podman-compose --version | Out-Null
    Write-Host "podman-compose is installed" -ForegroundColor Green
} catch {
    Write-Host "Error: podman-compose is not installed. Please install podman-compose first." -ForegroundColor Red
    Write-Host "Install with: pip install podman-compose" -ForegroundColor Yellow
    exit 1
}

# Check for GPU support
try {
    nvidia-smi | Out-Null
    Write-Host "NVIDIA GPU detected - GPU support will be available" -ForegroundColor Green
    $ComposeFile = "podman-compose.yml"
} catch {
    Write-Host "Warning: NVIDIA GPU not detected - using CPU-only mode" -ForegroundColor Yellow
    $ComposeFile = "podman-compose.cpu.yml"
}

# Build and start the containers
Write-Host "Building and starting containers..." -ForegroundColor Cyan
podman-compose -f $ComposeFile up --build -d

# Wait for services to be ready
Write-Host "Waiting for services to start..." -ForegroundColor Cyan
Start-Sleep -Seconds 10

# Check service health
Write-Host "Checking service health..." -ForegroundColor Cyan
podman-compose -f $ComposeFile ps

# Show logs for a few seconds
Write-Host "Recent logs:" -ForegroundColor Cyan
podman-compose -f $ComposeFile logs --tail=20

Write-Host ""
Write-Host "Space Debris Tracking System is now running!" -ForegroundColor Green
Write-Host "Access the application at: http://localhost:5000" -ForegroundColor Cyan
Write-Host "Redis is available at: localhost:6379" -ForegroundColor Cyan
Write-Host ""
Write-Host "To view logs: podman-compose -f $ComposeFile logs -f" -ForegroundColor Yellow
Write-Host "To stop: podman-compose -f $ComposeFile down" -ForegroundColor Yellow
