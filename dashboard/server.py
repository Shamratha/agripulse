"""AgriPulse dashboard server. Run:  uvicorn dashboard.server:app --port 8010"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent

app = FastAPI(title="AgriPulse")
app.mount("/outputs", StaticFiles(directory=ROOT / "outputs"), name="outputs")


@app.get("/")
def index():
    return FileResponse(ROOT / "dashboard" / "static" / "index.html")
