"""AgriPulse dashboard server. Run:  uvicorn dashboard.server:app --port 8010

Serves the map overlays + summary.json from an outputs directory. Locally that
is outputs/ (written by run_pipeline.py). On a hosting service with no GEE
access, set AGRIPULSE_OUTPUTS=demo_outputs to serve the committed real-data
snapshot instead.
"""

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent


def _outputs_dir():
    env = os.environ.get("AGRIPULSE_OUTPUTS")
    if env:
        return ROOT / env
    live = ROOT / "outputs"
    if (live / "summary.json").exists():
        return live
    return ROOT / "demo_outputs"  # committed real-data snapshot fallback


OUT = _outputs_dir()

app = FastAPI(title="AgriPulse")
app.mount("/outputs", StaticFiles(directory=OUT), name="outputs")


@app.get("/")
def index():
    return FileResponse(ROOT / "dashboard" / "static" / "index.html")
