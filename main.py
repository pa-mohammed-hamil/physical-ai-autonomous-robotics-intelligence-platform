"""
Physical AI & Autonomous Robotics Intelligence Platform
Main FastAPI Application - Simplified Version
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = PROJECT_ROOT / "frontend"

# FastAPI Application
app = FastAPI(
    title="Physical AI Robotics Platform",
    description="Autonomous Robotics Intelligence Platform with AR/VR, Video Streaming & AI",
    version="2.7.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Static Directories
for dirname in ["css", "js", "assets", "images", "fonts"]:
    directory = FRONTEND_DIR / dirname
    if directory.is_dir():
        app.mount(f"/{dirname}", StaticFiles(directory=str(directory)), name=dirname)
        logger.info(f"✓ Mounted /{dirname}")

# Home Page - Platform Navigation Hub
@app.get("/", include_in_schema=False)
async def home():
    """Serve the main platform home/navigation page"""
    platform_home = FRONTEND_DIR / "platform-home.html"
    if platform_home.is_file():
        return FileResponse(path=str(platform_home))
    
    # Fallback to index.html if platform-home doesn't exist
    index_html = FRONTEND_DIR / "index.html"
    if index_html.is_file():
        return FileResponse(path=str(index_html))
    
    return {"message": "Physical AI Platform API", "docs": "/api/docs"}

# Direct access to original dashboard
@app.get("/dashboard", include_in_schema=False)
@app.get("/index.html", include_in_schema=False)
async def dashboard_page():
    """Serve the original Aether dashboard"""
    index_html = FRONTEND_DIR / "index.html"
    if index_html.is_file():
        return FileResponse(path=str(index_html))
    raise HTTPException(status_code=404, detail="Dashboard not found")

# Frontend Pages - Multiple route patterns for flexibility
@app.get("/pages/{page_name}", include_in_schema=False)
@app.get("/frontend/{page_name}", include_in_schema=False)
@app.get("/{page_name}.html", include_in_schema=False)
async def frontend_page(page_name: str):
    """Serve frontend HTML pages with flexible routing"""
    # Remove .html extension if present
    page_name = page_name.replace('.html', '')
    
    # Security check
    if not page_name or page_name.startswith('.') or '/' in page_name or '\\' in page_name:
        raise HTTPException(status_code=400, detail="Invalid page name")
    
    html_file = FRONTEND_DIR / f"{page_name}.html"
    if not html_file.is_file():
        raise HTTPException(status_code=404, detail=f"Page not found: {page_name}")
    
    return FileResponse(path=str(html_file))

# Health Endpoints
@app.get("/api/v1/health")
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "application": "Physical AI Robotics Platform",
        "version": "2.7.0",
        "features": [
            "AR/VR Integration",
            "Video Streaming & Tutorials",
            "3D Robot Viewer",
            "AI Assistant",
            "Alert System",
            "Code Generation",
            "Image Damage Assessment"
        ]
    }

# Register API Routers
logger.info("Registering API routers...")

try:
    from app.api.v1.video_streaming import router as video_router
    app.include_router(video_router, prefix="/api/v1")
    logger.info("✓ Video Streaming router registered")
except Exception as e:
    logger.warning(f"✗ Video Streaming router: {e}")

try:
    from app.api.v1.ar_vr import router as ar_vr_router
    app.include_router(ar_vr_router, prefix="/api/v1")
    logger.info("✓ AR/VR router registered")
except Exception as e:
    logger.warning(f"✗ AR/VR router: {e}")

try:
    from app.api.v1.robot_3d_viewer import router as robot_3d_router
    app.include_router(robot_3d_router, prefix="/api/v1")
    logger.info("✓ 3D Robot Viewer router registered")
except Exception as e:
    logger.warning(f"✗ 3D Robot Viewer router: {e}")

try:
    from app.api.v1.ai_assistant import router as ai_assistant_router
    app.include_router(ai_assistant_router, prefix="/api/v1")
    logger.info("✓ AI Assistant router registered")
except Exception as e:
    logger.warning(f"✗ AI Assistant router: {e}")

try:
    from app.api.v1.alerts import router as alerts_router
    app.include_router(alerts_router, prefix="/api/v1")
    logger.info("✓ Alerts router registered")
except Exception as e:
    logger.warning(f"✗ Alerts router: {e}")

try:
    from app.api.v1.image_damage import router as image_damage_router
    app.include_router(image_damage_router, prefix="/api/v1")
    logger.info("✓ Image Damage Assessment router registered")
except Exception as e:
    logger.warning(f"✗ Image Damage router: {e}")

try:
    from app.api.v1.copilot_code import router as copilot_router
    app.include_router(copilot_router, prefix="/api/v1")
    logger.info("✓ Copilot Code Generation router registered")
except Exception as e:
    logger.warning(f"✗ Copilot router: {e}")

# Dashboard
@app.get("/api/v1/dashboard")
async def dashboard():
    """Return dashboard data with all available pages"""
    return {
        "status": "online",
        "platform": "Physical AI Robotics Platform",
        "version": "2.7.0",
        "features": [
            "AR/VR Integration",
            "Video Streaming & Tutorials",
            "3D Robot Viewer",
            "AI Assistant",
            "Alert System",
            "Code Generation",
            "Image Damage Assessment"
        ],
        "pages": {
            "navigation": {
                "platform-home": "/",
                "original-dashboard": "/dashboard",
                "api-docs": "/api/docs"
            },
            "video": {
                "video-streaming": "/frontend/video-streaming",
                "tutorials": "/frontend/video-streaming#tutorials"
            },
            "ar-vr": {
                "ar-vr-viewer": "/frontend/ar-vr-viewer",
                "vr-environments": "/frontend/ar-vr-viewer#environments"
            },
            "robot": {
                "3d-viewer": "/frontend/robot-3d-viewer",
                "robot-details": "/frontend/robot-details"
            },
            "ai": {
                "ai-assistant": "/frontend/ai-support-assistant",
                "chatbot": "/frontend/ai-support-assistant#chat"
            },
            "alerts": {
                "alerts-dashboard": "/frontend/robot-alerts-pro-dashboard",
                "basic-alerts": "/frontend/robot-alerts-dashboard"
            },
            "assessment": {
                "damage-assessment": "/frontend/damage-assessment"
            }
        }
    }

# Get all available pages
@app.get("/api/v1/pages")
async def list_pages():
    """List all available frontend pages"""
    pages = []
    if FRONTEND_DIR.is_dir():
        for html_file in FRONTEND_DIR.glob("*.html"):
            pages.append({
                "name": html_file.stem,
                "filename": html_file.name,
                "url": f"/frontend/{html_file.stem}",
                "size_kb": html_file.stat().st_size / 1024
            })
    
    return {
        "total": len(pages),
        "pages": sorted(pages, key=lambda x: x['name'])
    }

# WebSocket Demo
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        await websocket.send_json({"type": "connected", "message": "WebSocket connected"})
        while True:
            message = await websocket.receive_text()
            await websocket.send_json({"type": "echo", "message": message})
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")

# Startup Event
@app.on_event("startup")
async def startup_event():
    logger.info("=" * 60)
    logger.info("Physical AI Robotics Platform v2.7.0")
    logger.info("=" * 60)
    logger.info("Server running at: http://localhost:8000")
    logger.info("API Docs: http://localhost:8000/api/docs")
    logger.info("")
    logger.info("🎥 Video Streaming: http://localhost:8000/frontend/video-streaming")
    logger.info("🥽 AR/VR Viewer: http://localhost:8000/frontend/ar-vr-viewer")
    logger.info("🤖 3D Robot Viewer: http://localhost:8000/frontend/robot-3d-viewer")
    logger.info("=" * 60)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
