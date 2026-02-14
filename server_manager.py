"""LLM Manager — Server process lifecycle manager for llama-server."""

import asyncio
import os
import signal
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import httpx


class ServerState(str, Enum):
    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class ServerInfo:
    state: ServerState = ServerState.IDLE
    model_path: Optional[str] = None
    model_name: Optional[str] = None
    pid: Optional[int] = None
    port: int = 8081
    started_at: Optional[float] = None
    error_message: Optional[str] = None
    n_gpu_layers: int = 18
    ctx_size: int = 4096

    def to_dict(self) -> dict:
        return {
            "state": self.state.value,
            "model_path": self.model_path,
            "model_name": self.model_name,
            "pid": self.pid,
            "port": self.port,
            "started_at": self.started_at,
            "uptime_seconds": round(time.time() - self.started_at, 1) if self.started_at else None,
            "error_message": self.error_message,
            "n_gpu_layers": self.n_gpu_layers,
            "ctx_size": self.ctx_size,
        }


# Path to llama-server binary
LLAMA_SERVER_BIN = os.environ.get(
    "LLAMA_SERVER_BIN",
    os.path.expanduser("~/workspace/llama.cpp/build/bin/llama-server"),
)

LLAMA_SERVER_PORT = int(os.environ.get("LLAMA_SERVER_PORT", "8081"))
HEALTH_CHECK_TIMEOUT = 120  # seconds to wait for model loading
HEALTH_CHECK_INTERVAL = 1.0  # seconds between health checks


class ServerManager:
    """Singleton manager for a llama-server subprocess."""

    _instance: Optional["ServerManager"] = None

    def __new__(cls) -> "ServerManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._process: Optional[asyncio.subprocess.Process] = None
        self._info = ServerInfo(port=LLAMA_SERVER_PORT)
        self._stdout_task: Optional[asyncio.Task] = None
        self._log_lines: list[str] = []

    @property
    def info(self) -> ServerInfo:
        return self._info

    @property
    def logs(self) -> list[str]:
        return self._log_lines[-200:]  # last 200 lines

    async def start(
        self,
        model_path: str,
        ctx_size: int = 4096,
        n_gpu_layers: int = 18,
    ) -> ServerInfo:
        """Start llama-server with the given model."""
        if self._info.state in (ServerState.RUNNING, ServerState.STARTING):
            await self.stop()

        self._info = ServerInfo(
            state=ServerState.STARTING,
            model_path=model_path,
            model_name=os.path.basename(model_path),
            port=LLAMA_SERVER_PORT,
            n_gpu_layers=n_gpu_layers,
            ctx_size=ctx_size,
        )
        self._log_lines = []

        # Auto-detect mmproj (vision projector) file
        mmproj_arg = []
        base_name = os.path.splitext(model_path)[0]
        model_dir = os.path.dirname(model_path)
        
        candidates = [
            f"{base_name}.mmproj",
            os.path.join(model_dir, "mmproj-model-f16.gguf"),
            os.path.join(model_dir, f"mmproj-{os.path.basename(base_name)}.gguf"),
        ]

        for cand in candidates:
            if os.path.exists(cand):
                mmproj_arg = ["--mmproj", cand]
                # Vision models need extra VRAM for the image encoder;
                # cap GPU layers to avoid OOM on 8 GB cards.
                n_gpu_layers = min(n_gpu_layers, 10)
                break

        cmd = [
            LLAMA_SERVER_BIN,
            "--model", model_path,
            *mmproj_arg,
            "--port", str(LLAMA_SERVER_PORT),
            "--host", "127.0.0.1",
            "--ctx-size", str(ctx_size),
            "--n-gpu-layers", str(n_gpu_layers),
            "--flash-attn", "on",
        ]

        try:
            self._process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            self._info.pid = self._process.pid

            # Start reading stdout in background
            self._stdout_task = asyncio.create_task(self._read_stdout())

            # Wait for health check
            healthy = await self._wait_for_health()
            if healthy:
                self._info.state = ServerState.RUNNING
                self._info.started_at = time.time()
            else:
                self._info.state = ServerState.ERROR
                self._info.error_message = "Server failed to start within timeout"
                await self._kill_process()

        except Exception as e:
            self._info.state = ServerState.ERROR
            self._info.error_message = str(e)

        return self._info

    async def stop(self) -> ServerInfo:
        """Stop the running llama-server."""
        if self._process is None or self._info.state == ServerState.IDLE:
            return self._info

        self._info.state = ServerState.STOPPING

        await self._kill_process()

        self._info.state = ServerState.IDLE
        self._info.pid = None
        self._info.started_at = None
        self._info.error_message = None
        self._process = None

        return self._info

    async def _kill_process(self):
        """Gracefully terminate, then force kill if needed."""
        if self._process is None:
            return

        try:
            self._process.send_signal(signal.SIGTERM)
            try:
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._process.kill()
                await self._process.wait()
        except ProcessLookupError:
            pass  # Already exited

        if self._stdout_task and not self._stdout_task.done():
            self._stdout_task.cancel()
            try:
                await self._stdout_task
            except asyncio.CancelledError:
                pass

    async def _read_stdout(self):
        """Read stdout from the subprocess and store log lines."""
        if self._process is None or self._process.stdout is None:
            return
        try:
            async for line in self._process.stdout:
                decoded = line.decode("utf-8", errors="replace").rstrip()
                if decoded:
                    self._log_lines.append(decoded)
        except Exception:
            pass

    async def _wait_for_health(self) -> bool:
        """Poll the /health endpoint until the server is ready."""
        url = f"http://127.0.0.1:{LLAMA_SERVER_PORT}/health"
        deadline = time.time() + HEALTH_CHECK_TIMEOUT

        async with httpx.AsyncClient() as client:
            while time.time() < deadline:
                # Check if process has already died
                if self._process and self._process.returncode is not None:
                    return False
                try:
                    resp = await client.get(url, timeout=2.0)
                    if resp.status_code == 200:
                        data = resp.json()
                        if data.get("status") == "ok":
                            return True
                except (httpx.ConnectError, httpx.ReadError, httpx.TimeoutException):
                    pass
                await asyncio.sleep(HEALTH_CHECK_INTERVAL)

        return False

    async def shutdown(self):
        """Clean up on application shutdown."""
        if self._info.state in (ServerState.RUNNING, ServerState.STARTING):
            await self.stop()
