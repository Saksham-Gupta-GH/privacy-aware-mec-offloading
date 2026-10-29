import time
import os
import math
import numpy as np
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import Optional

# Retrieve API key from environment
API_KEY = os.getenv("AZURE_API_KEY", "prism_sec_healthcare_mec_2026")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

app = FastAPI(
    title="PRISM-MEC Azure Cloud Compute Worker",
    description="Cloud tier worker node for Mobile Edge Computing offloading testbed",
    version="1.0.0"
)

def verify_api_key(key: str = Depends(api_key_header)):
    if not key or key != API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Forbidden: Invalid or missing X-API-Key header"
        )
    return key


class TaskExecutionRequest(BaseModel):
    task_id: int
    cpu_cycles_mflop: float
    payload_size_kb: Optional[float] = 10.0
    payload_data: Optional[str] = None


class CalibrationRequest(BaseModel):
    benchmark_mflop: float = 100.0
    payload_kb: float = 50.0


@app.get("/health")
def health_check():
    """Public health check endpoint."""
    return {
        "status": "healthy",
        "service": "PRISM-MEC Azure Cloud Worker",
        "tier": "cloud",
        "timestamp": time.time()
    }


@app.post("/execute")
def execute_task(req: TaskExecutionRequest, authorized: str = Depends(verify_api_key)):
    """
    Executes a healthcare computation workload (simulating ECG/MRI filtering via matrix ops)
    and returns exact execution timing.
    """
    start_time = time.perf_counter()

    # Synthetic compute load: scale matrix dimension proportionally to MFLOPs
    # A matrix multiplication of size N x N requires ~ 2 * N^3 operations
    n_dim = int(math.pow((req.cpu_cycles_mflop * 1e6) / 2.0, 1.0 / 3.0))
    n_dim = max(50, min(n_dim, 600))  # Bound between 50 and 600 for safe execution

    # Perform real matrix operations
    a = np.random.rand(n_dim, n_dim)
    b = np.random.rand(n_dim, n_dim)
    _ = np.dot(a, b)

    exec_time_ms = (time.perf_counter() - start_time) * 1000.0

    return {
        "task_id": req.task_id,
        "cpu_cycles_mflop": req.cpu_cycles_mflop,
        "matrix_dim": n_dim,
        "exec_time_ms": round(exec_time_ms, 3),
        "status": "completed",
        "processed_by": "Azure-Cloud-Tier"
    }


@app.post("/benchmark")
def run_calibration(req: CalibrationRequest, authorized: str = Depends(verify_api_key)):
    """
    Calibration endpoint for empirical system identification.
    """
    start_time = time.perf_counter()
    n_dim = int(math.pow((req.benchmark_mflop * 1e6) / 2.0, 1.0 / 3.0))
    n_dim = max(50, min(n_dim, 800))

    a = np.random.rand(n_dim, n_dim)
    b = np.random.rand(n_dim, n_dim)
    _ = np.dot(a, b)
    compute_time_ms = (time.perf_counter() - start_time) * 1000.0

    return {
        "benchmark_mflop": req.benchmark_mflop,
        "payload_received_kb": req.payload_kb,
        "compute_time_ms": round(compute_time_ms, 3),
        "server_time": time.time()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("AZURE_API_PORT", 8000)))
