# Container for the AgriPulse dashboard (Fly.io / any container host).
# Serves the committed real-data snapshot in demo_outputs/ — no GEE, no ML at
# runtime, so the image is tiny and starts fast.
FROM python:3.13-slim

WORKDIR /app

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY dashboard/ dashboard/
COPY demo_outputs/ demo_outputs/

ENV AGRIPULSE_OUTPUTS=demo_outputs
EXPOSE 8080

CMD ["uvicorn", "dashboard.server:app", "--host", "0.0.0.0", "--port", "8080"]
