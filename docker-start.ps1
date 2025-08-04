# PowerShell Docker startup script for Space Debris Tracking System

Write-Host "Starting Space Debris Tracking System with Docker..." -ForegroundColor Green

# Check if Docker is running
try {
    docker info | Out-Null
    Write-Host "Docker is running" -ForegroundColor Green
} catch {
    Write-Host "Error: Docker is not running. Please start Docker first." -ForegroundColor Red
    exit 1
}

# Check if nvidia-docker runtime is available (for GPU support)
$dockerInfo = docker info 2>$null
if ($dockerInfo -match "nvidia") {
    Write-Host "NVIDIA Docker runtime detected - GPU support will be available" -ForegroundColor Green
    $ComposeFile = "docker-compose.yml"
} else {
    Write-Host "Warning: NVIDIA Docker runtime not detected - using CPU-only mode" -ForegroundColor Yellow
    $ComposeFile = "docker-compose.cpu.yml"
}

# Build and start the containers
Write-Host "Building and starting containers..." -ForegroundColor Cyan
docker-compose -f $ComposeFile up --build -d

# Wait for services to be ready
Write-Host "Waiting for services to start..." -ForegroundColor Cyan
Start-Sleep -Seconds 10

# Check service health
Write-Host "Checking service health..." -ForegroundColor Cyan
docker-compose -f $ComposeFile ps

# Show logs for a few seconds
Write-Host "Recent logs:" -ForegroundColor Cyan
docker-compose -f $ComposeFile logs --tail=20

Write-Host ""
Write-Host "Space Debris Tracking System is now running!" -ForegroundColor Green
Write-Host "Access the application at: http://localhost:5000" -ForegroundColor Cyan
Write-Host "Redis is available at: localhost:6379" -ForegroundColor Cyan
Write-Host ""
Write-Host "To view logs: docker-compose -f $ComposeFile logs -f" -ForegroundColor Yellow
Write-Host "To stop: docker-compose -f $ComposeFile down" -ForegroundColor Yellow
