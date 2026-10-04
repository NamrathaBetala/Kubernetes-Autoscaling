from fastapi import FastAPI, UploadFile, File
import httpx
import time
from prometheus_client import Counter, Histogram, Gauge, make_asgi_app

app = FastAPI()

CLASSIFIER_URL = "http://ml-inference-service:8080/predict"

# Prometheus Metrics
REQUESTS_TOTAL = Counter(
    "dispatcher_requests_total",
    "Total number of requests received by the dispatcher"
)

REQUEST_LATENCY = Histogram(
    "dispatcher_latency_seconds",
    "Latency of dispatcher requests"
)

ACTIVE_REQUESTS = Gauge(
    "dispatcher_active_requests",
    "Number of requests currently being processed"
)

# Expose metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.get("/")
async def root():
    return {"message": "Dispatcher is running"}


@app.post("/predict")
async def dispatch(file: UploadFile = File(...)):
    REQUESTS_TOTAL.inc()
    ACTIVE_REQUESTS.inc()
    start_time = time.time()

    try:
        file_bytes = await file.read()
        timeout = httpx.Timeout(10.0)  # 10s timeout

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                CLASSIFIER_URL,
                files={
                    "file": (
                        file.filename,
                        file_bytes,
                        file.content_type
                    )
                }
            )

        # Successful response
        if response.status_code == 200:
            return response.json()

        # Inference returned an error
        return {
            "error": response.text,
            "status": response.status_code
        }

    except Exception as e:
        # Prevent dispatcher crash
        return {"error": str(e)}

    finally:
        REQUEST_LATENCY.observe(time.time() - start_time)
        ACTIVE_REQUESTS.dec()
