"""
CSIDC Land Watch - FastAPI Backend
Main application server for automated land monitoring system
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import os
import shutil
from pathlib import Path
from typing import Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="CSIDC Land Watch API",
    description="Automated Land Monitoring System for Chhattisgarh State Industrial Development Corporation",
    version="1.0.0"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create necessary directories
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

PROCESSED_DIR = Path("processed")
PROCESSED_DIR.mkdir(exist_ok=True)

# Lazy import and initialize modules
aligner = None
detector = None

def get_aligner():
    global aligner
    if aligner is None:
        try:
            from align import ImageAligner
            aligner = ImageAligner()
        except Exception as e:
            logger.error(f"Failed to load ImageAligner: {e}")
            raise HTTPException(status_code=500, detail="Image alignment module not available")
    return aligner

def get_detector():
    global detector
    if detector is None:
        try:
            from detector import ChangeDetector
            detector = ChangeDetector()
        except Exception as e:
            logger.error(f"Failed to load ChangeDetector: {e}")
            raise HTTPException(status_code=500, detail="Change detection module not available")
    return detector

# Serve frontend
frontend_path = Path(__file__).parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")


@app.get("/")
async def root():
    """Serve the main dashboard"""
    return FileResponse(str(frontend_path / "index.html"))


@app.post("/upload-map")
async def upload_map(
    file: UploadFile = File(...),
    map_type: str = "legacy",
    lat_min: Optional[float] = None,
    lat_max: Optional[float] = None,
    lon_min: Optional[float] = None,
    lon_max: Optional[float] = None
):
    """
    Upload a legacy or current map image and align it to GPS coordinates
    
    Args:
        file: Image file (PNG/JPG)
        map_type: Type of map ("legacy" or "current")
        lat_min, lat_max, lon_min, lon_max: Bounding box coordinates
    
    Returns:
        JSON with aligned image path and metadata
    """
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Create filename with map type prefix
        filename = f"{map_type}_{file.filename}"
        file_path = UPLOAD_DIR / filename
        
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Uploaded {map_type} file: {filename}")
        
        # If coordinates provided, perform geo-rectification
        if all([lat_min, lat_max, lon_min, lon_max]):
            logger.info(f"Aligning image to bounds: ({lat_min}, {lon_min}) to ({lat_max}, {lon_max})")
            
            aligned_path = get_aligner().align_image(
                str(file_path),
                bounds={
                    "lat_min": lat_min,
                    "lat_max": lat_max,
                    "lon_min": lon_min,
                    "lon_max": lon_max
                }
            )
            
            return JSONResponse({
                "status": "success",
                "message": f"{map_type.capitalize()} map uploaded and aligned successfully",
                "map_type": map_type,
                "original_path": str(file_path),
                "aligned_path": aligned_path,
                "bounds": {
                    "lat_min": lat_min,
                    "lat_max": lat_max,
                    "lon_min": lon_min,
                    "lon_max": lon_max
                }
            })
        else:
            return JSONResponse({
                "status": "success",
                "message": f"{map_type.capitalize()} map uploaded successfully (no alignment performed)",
                "map_type": map_type,
                "original_path": str(file_path)
            })
    
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect-changes")
async def detect_changes(
    legacy_map: str,
    current_map: str
):
    """
    Detect changes between legacy and current satellite maps
    
    Args:
        legacy_map: Path to aligned legacy map
        current_map: Path to current satellite imagery
    
    Returns:
        JSON with detected changes and visualization
    """
    try:
        results = get_detector().detect_changes(legacy_map, current_map)
        
        return JSONResponse({
            "status": "success",
            "changes_detected": results["changes_detected"],
            "change_percentage": results["change_percentage"],
            "visualization_path": results["visualization_path"],
            "alerts": results["alerts"]
        })
    
    except Exception as e:
        logger.error(f"Error detecting changes: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "CSIDC Land Watch"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
