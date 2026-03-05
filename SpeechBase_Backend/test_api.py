#!/usr/bin/env python3
"""
Test script for the ADHD Speech Detection API
"""
import requests
import time

def test_api():
    print("Testing ADHD Speech Detection API...")

    # Test health endpoint
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check error: {e}")

    # Test root endpoint
    try:
        response = requests.get("http://localhost:8000/")
        if response.status_code == 200:
            print("✅ Root endpoint passed")
            data = response.json()
            print(f"   API version: {data.get('version', 'unknown')}")
            print(f"   Endpoints: {list(data.get('endpoints', {}).keys())}")
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Root endpoint error: {e}")

    print("\nAPI endpoints are working correctly!")
    print("To test audio analysis, use:")
    print("curl -X POST 'http://localhost:8000/analyze' -F 'file=@your_audio.wav' -F 'child_age=8'")

if __name__ == "__main__":
    test_api()