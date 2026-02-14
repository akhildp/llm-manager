"""LLM Manager — FastAPI entry point."""

import contextlib
import os
import sys

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from routers import models, server, chat
from server_manager import ServerManager


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    yield
    # Cleanup: stop llama-server if running
    manager = ServerManager()
    await manager.shutdown()


app = FastAPI(
    title="LLM Manager",
    description="Local LLM management interface for llama.cpp",
    version="1.0.0",
    lifespan=lifespan,
)

# Include API routers
app.include_router(models.router)
app.include_router(server.router)
app.include_router(chat.router)

# Serve static frontend files
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
    )
