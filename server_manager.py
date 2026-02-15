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
    is_multimodal: bool = False

    def to_dict(self) -> dict:
        return {
            "state": self.state.value,
            "model_path": self.model_path,
            "model_name": self.model_name,
            "pid": self.pid,
            "port": self.port,
            "started_at": self.started_at,
            "error_message": self.error_message,
            "n_gpu_layers": self.n_gpu_layers,
            "ctx_size": self.ctx_size,
            "is_multimodal": self.is_multimodal,
            "uptime_seconds": round(time.time() - self.started_at, 1) if self.started_at is not None else None,
        }


# Path to llama-server binary
LLAMA_SERVER_BIN = os.environ.get(
    "LLAMA_SERVER_BIN",
    os.path.expanduser("~/workspace/llama.cpp/build/bin/llama-server"),
)

LLAMA_SERVER_PORT = int(os.environ.get("LLAMA_SERVER_PORT", "8081"))
HEALTH_CHECK_TIMEOUT = 120  # seconds to wait for model loading
HEALTH_CHECK_INTERVAL = 1.0  # seconds between health checks


MODEL_OPTIMIZATIONS = {
    "nemotron": {
        "n_gpu_layers": 18, # Reduced to 18 for stability
        "ctx_size": 8192,   
    },
    "phi": {
        "n_gpu_layers": 10, # Reduced to 10 for stability
        "ctx_size": 2048,   # Summaries are small
    }
}

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

        # Apply model-specific optimizations
        model_name = os.path.basename(model_path)
        model_lower = model_name.lower()
        for key, opts in MODEL_OPTIMIZATIONS.items():
            if key in model_lower:
                n_gpu_layers = opts.get("n_gpu_layers", n_gpu_layers)
                ctx_size = opts.get("ctx_size", ctx_size)
                break

        self._info = ServerInfo(
            state=ServerState.STARTING,
            model_path=model_path,
            model_name=model_name,
            port=LLAMA_SERVER_PORT,
            n_gpu_layers=n_gpu_layers,
            ctx_size=ctx_size,
        )
        self._log_lines = []

        # Auto-detect mmproj (vision projector) file
        mmproj_arg = []
        base_name = os.path.splitext(model_path)[0]
        model_dir = os.path.dirname(model_path)
        
        # Specific candidates that include the model's base name
        candidates = [
            f"{base_name}.mmproj",
            os.path.join(model_dir, f"mmproj-{os.path.basename(base_name)}.gguf"),
            os.path.join(model_dir, f"{os.path.basename(base_name)}-mmproj.gguf"),
        ]

        # Only allow generic projectors if the model name indicates vision capability
        vision_keywords = ["llava", "vision", "moondream", "qwen-vl", "internlm-xcomposer", "obsidian"]
        if any(k in model_lower for k in vision_keywords):
            candidates.append(os.path.join(model_dir, "mmproj-model-f16.gguf"))

        for cand in candidates:
            if os.path.exists(cand):
                mmproj_arg = ["--mmproj", cand]
                self._info.is_multimodal = True
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

    async def infer(self, prompt: str, **kwargs) -> dict:
        """Run inference on the main model. Returns {content, t_s}."""
        if self._info.state != ServerState.RUNNING:
            return {"content": "Error: Main model is not running.", "t_s": 0}

        payload = {
            "prompt": f"<|user|>\n{prompt}<|end|>\n<|assistant|>",
            "n_predict": 1024,
            "temperature": 0.3,
            "stop": ["<|end|>", "<|user|>", "<|assistant|>"]
        }
        payload.update(kwargs)

        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                resp = await client.post(
                    f"http://127.0.0.1:{LLAMA_SERVER_PORT}/completion",
                    json=payload
                )
                resp.raise_for_status()
                data = resp.json()
                content = data.get("content", "").strip()
                t_s = data.get("timings", {}).get("predicted_per_second", 0)
                return {"content": content, "t_s": t_s}
            except Exception as e:
                return {"content": f"Error during main model inference: {str(e)}", "t_s": 0}

    async def stop(self) -> None:
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
            
            
class UtilityServerManager:
    """Manager for a secondary utility model (Phi-3 Mini) on port 8082."""

    _instance: Optional["UtilityServerManager"] = None
    
    # Phi-3 Mini 4k Instruct Q4
    MODEL_URL = "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf"
    MODEL_FILENAME = "Phi-3-mini-4k-instruct-q4.gguf"
    PORT = 8082

    def __new__(cls) -> "UtilityServerManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._process: Optional[asyncio.subprocess.Process] = None
        self._model_path: Optional[str] = None
        self._lock = asyncio.Lock()
        self.is_ready = False

    async def ensure_model(self):
        """Download text utility model if missing."""
        models_dir = os.path.expanduser("~/workspace/llama.cpp/models")
        os.makedirs(models_dir, exist_ok=True)
        
        self._model_path = os.path.join(models_dir, self.MODEL_FILENAME)
        
        if os.path.exists(self._model_path):
            return

        print(f"[UTILITY] Downloading {self.MODEL_FILENAME}...")
        async with httpx.AsyncClient(follow_redirects=True, timeout=600.0) as client:
            async with client.stream("GET", self.MODEL_URL) as response:
                response.raise_for_status()
                with open(self._model_path, "wb") as f:
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        f.write(chunk)
        print(f"[UTILITY] Download complete: {self._model_path}")

    async def _start_impl(self):
        """Internal start implementation (no lock)."""
        await self.ensure_model()
        
        if self._process and self._process.returncode is None:
            return

        cmd = [
            LLAMA_SERVER_BIN,
            "--model", self._model_path,
            "--port", str(self.PORT),
            "--ctx-size", "2048",
            "--n-gpu-layers", "10", # Reduced to 10 for stability
            "--flash-attn", "on",
            "--n-predict", "1024", 
            "--threads", "4"
        ]

        log_path = os.path.expanduser("~/workspace/llm-manager/utility_server.log")
        print(f"[UTILITY] Starting server on port {self.PORT}, logs: {log_path}")
        
        self._process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=open(log_path, "w"),
        )
        
        # Wait for health check
        for _ in range(60):
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(f"http://127.0.0.1:{self.PORT}/health", timeout=1.0)
                    if resp.status_code == 200:
                        self.is_ready = True
                        print(f"[UTILITY] Server ready on port {self.PORT}")
                        return
            except Exception:
                await asyncio.sleep(1)
        
        print("[UTILITY] Failed to start server within timeout.")

    async def start(self):
        """Start the utility server (thread-safe)."""
        async with self._lock:
            await self._start_impl()

    async def _stop_impl(self):
        """Internal stop implementation (no lock)."""
        if self._process:
            if self._process.returncode is None:
                self._process.terminate()
                try:
                    await asyncio.wait_for(self._process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    self._process.kill()
            self._process = None
            self.is_ready = False

    async def stop(self):
        """Stop the utility server (thread-safe)."""
        async with self._lock:
            await self._stop_impl()

    async def infer(self, prompt: str, **kwargs) -> dict:
        """Run inference on the utility model. Returns {content, t_s}."""
        async with self._lock:
            if not self.is_ready:
                await self._start_impl()
            
            payload = {
                "prompt": f"<|user|>\n{prompt}<|end|>\n<|assistant|>",
                "n_predict": 1024,
                "temperature": 0.3,
                "stop": ["<|end|>", "<|user|>", "<|assistant|>"]
            }
            # Update with any overrides
            payload.update(kwargs)

            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    f"http://127.0.0.1:{self.PORT}/completion",
                    json=payload
                )
                resp.raise_for_status()
            data = resp.json()
            content = data.get("content", "").strip()
            t_s = data.get("timings", {}).get("predicted_per_second", 0)
            return {"content": content, "t_s": t_s}
