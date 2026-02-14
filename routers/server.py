"""API router for llama-server lifecycle management."""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

from server_manager import ServerManager

router = APIRouter(prefix="/api/server", tags=["server"])


class StartRequest(BaseModel):
    model_path: str
    ctx_size: int = 4096
    n_gpu_layers: int = 18


@router.post("/start")
async def start_server(req: StartRequest):
    """Start llama-server with the specified model."""
    from routers.models import _load_model_dirs
    from pathlib import Path
    import os

    # Security: Validate model_path is within allowed directories
    model_path = os.path.abspath(req.model_path)
    allowed_dirs = [os.path.abspath(d) for d in _load_model_dirs()]
    
    is_allowed = False
    for d in allowed_dirs:
        if model_path.startswith(d):
            is_allowed = True
            break
            
    if not is_allowed:
        return {"error": "Invalid model path. Path must be within configured model directories.", "status": "error"}

    if not os.path.exists(model_path):
        return {"error": f"Model file not found: {req.model_path}", "status": "error"}

    manager = ServerManager()
    info = await manager.start(
        model_path=model_path,
        ctx_size=req.ctx_size,
        n_gpu_layers=req.n_gpu_layers,
    )
    return info.to_dict()


@router.post("/stop")
async def stop_server():
    """Stop the running llama-server."""
    manager = ServerManager()
    info = await manager.stop()
    return info.to_dict()


@router.get("/status")
async def server_status():
    """Get current server status."""
    manager = ServerManager()
    return manager.info.to_dict()


@router.get("/logs")
async def server_logs():
    """Get recent server log lines."""
    manager = ServerManager()
    return {"logs": manager.logs}
