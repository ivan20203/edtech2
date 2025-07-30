#!/usr/bin/env python3
"""
Minimal FastAPI server for MoonCast podcast generation.
Handles async jobs with status tracking.
"""

import os
import sys
import json
import tempfile
import base64
import time
import threading
from pathlib import Path
from typing import Dict, Any
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the PodcastGenerator class
from MoonCast_seed import PodcastGenerator

app = FastAPI(title="MoonCast Podcast Generator")

class PodcastRequest(BaseModel):
    topic: str
    duration: int = 1
    seed: int = None

# Global storage for jobs
jobs = {}
generator = None

def initialize_generator():
    """Initialize the podcast generator"""
    global generator
    print("Initializing Podcast Generator...")
    try:
        generator = PodcastGenerator()
        print("✅ Podcast Generator initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize generator: {e}")
        generator = None

def generate_podcast_background(job_id: str, topic: str, duration: int, seed: int):
    """Background task for podcast generation"""
    try:
        jobs[job_id]["status"] = "running"
        
        # Generate output filename
        timestamp = int(time.time())
        output_filename = f"podcast_{timestamp}.wav"
        output_path = output_filename  # Save in current directory
        
        # Generate the podcast
        result_path = generator.generate_podcast(
            topic=topic,
            output_path=output_path,
            duration_minutes=duration,
            use_input_file=False,
            save_script=False
        )
        
        # Read the generated audio file
        with open(result_path, 'rb') as f:
            audio_data = f.read()
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        
        jobs[job_id] = {
            "status": "completed",
            "audioData": audio_base64,
            "fileName": os.path.basename(result_path),
            "fileSize": len(audio_data)
        }
        
    except Exception as e:
        jobs[job_id] = {
            "status": "failed",
            "error": str(e)
        }

@app.on_event("startup")
async def startup_event():
    """Initialize the generator when the server starts"""
    thread = threading.Thread(target=initialize_generator)
    thread.start()

@app.post("/generate")
async def start_generation(request: PodcastRequest):
    """Start podcast generation (returns job ID)"""
    if not generator:
        raise HTTPException(status_code=503, detail="Generator not ready yet")
    
    job_id = f"job_{int(time.time())}_{hash(request.topic) % 10000}"
    
    jobs[job_id] = {"status": "queued"}
    
    # Start generation in background
    thread = threading.Thread(
        target=generate_podcast_background,
        args=(job_id, request.topic, request.duration, request.seed)
    )
    thread.start()
    
    return {"job_id": job_id, "status": "started"}

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """Get job status and result"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return jobs[job_id]

@app.get("/files")
async def list_files():
    """List all generated audio files"""
    import glob
    files = []
    
    # Get all .wav files in current directory
    wav_files = glob.glob("podcast_*.wav")
    
    for file_path in wav_files:
        filename = os.path.basename(file_path)
        stat = os.stat(file_path)
        files.append({
            "name": filename,
            "size": stat.st_size,
            "created": stat.st_ctime,
            "url": f"/download/{filename}"
        })
    
    return {"files": files}

@app.get("/download/{filename}")
async def download_file(filename: str):
    """Download a generated audio file"""
    file_path = filename  # Use current directory
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    from fastapi.responses import FileResponse
    return FileResponse(file_path, filename=filename)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8888) 