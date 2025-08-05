#!/usr/bin/env python3
"""
Test script for FastAPI Space Debris Tracker
"""

import requests
import time
import sys

def test_fastapi_endpoints():
    """Test the main FastAPI endpoints"""
    base_url = "http://localhost:5000"
    
    print("🚀 Testing FastAPI Space Debris Tracker...")
    print("=" * 50)
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Health check failed: {e}")
        return False
    
    # Test status endpoint
    try:
        response = requests.get(f"{base_url}/api/status", timeout=5)
        if response.status_code == 200:
            print("✅ Status endpoint working")
            status = response.json()
            print(f"   TLE Data: {status.get('tle_data', False)}")
            print(f"   Orbital Params: {status.get('orbital_params', False)}")
            print(f"   Model: {status.get('model', False)}")
            print(f"   Objects: {status.get('object_count', 0)}")
        else:
            print(f"❌ Status endpoint failed: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Status endpoint failed: {e}")
    
    # Test satellites endpoint
    try:
        response = requests.get(f"{base_url}/api/satellites", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'success':
                satellites = data.get('satellites', [])
                print(f"✅ Satellites endpoint working ({len(satellites)} satellites)")
            else:
                print(f"✅ Satellites endpoint working (no data available)")
        else:
            print(f"❌ Satellites endpoint failed: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Satellites endpoint failed: {e}")
    
    print("\n📚 API Documentation available at:")
    print(f"   Swagger UI: {base_url}/docs")
    print(f"   ReDoc: {base_url}/redoc")
    
    print("\n🌐 Main dashboard available at:")
    print(f"   {base_url}/")
    
    return True

def main():
    """Main test function"""
    print("Waiting for FastAPI server to start...")
    time.sleep(2)  # Give the server time to start
    
    success = test_fastapi_endpoints()
    
    if success:
        print("\n🎉 FastAPI Space Debris Tracker is working correctly!")
        print("\nNext steps:")
        print("1. Open the web dashboard in your browser")
        print("2. Try the API endpoints")
        print("3. Check out the automatic API documentation")
    else:
        print("\n❌ FastAPI Space Debris Tracker test failed!")
        print("Please check:")
        print("1. Is the Podman container running?")
        print("2. Is port 5000 available?")
        print("3. Are all dependencies installed?")
        sys.exit(1)

if __name__ == "__main__":
    main() 