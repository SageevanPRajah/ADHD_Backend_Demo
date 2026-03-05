#!/usr/bin/env python3
"""
Script to start both backend and frontend servers
"""
import subprocess
import sys
import time
import os

def start_backend():
    """Start the FastAPI backend server"""
    print("🚀 Starting FastAPI backend server...")
    backend_cmd = [sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000", "--reload"]
    return subprocess.Popen(backend_cmd, cwd=os.getcwd())

def start_frontend():
    """Start the React frontend server"""
    print("🎨 Starting React frontend server...")
    frontend_cmd = ["npm", "start"]
    return subprocess.Popen(frontend_cmd, cwd=os.path.join(os.getcwd(), "frontend"), shell=True)

def main():
    print("🎯 ADHD Speech Detection - Full Application Startup")
    print("=" * 50)

    # Start backend
    backend_process = start_backend()
    time.sleep(3)  # Wait for backend to start

    # Start frontend
    frontend_process = start_frontend()
    time.sleep(5)  # Wait for frontend to start

    print("\n✅ Application started successfully!")
    print("📱 Frontend: http://localhost:3000")
    print("🔌 API: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop both servers...")

    try:
        # Wait for both processes
        backend_process.wait()
        frontend_process.wait()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down servers...")
        backend_process.terminate()
        frontend_process.terminate()
        backend_process.wait()
        frontend_process.wait()
        print("✅ Servers stopped successfully")

if __name__ == "__main__":
    main()