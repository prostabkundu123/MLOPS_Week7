from fastapi import FastAPI, Request, HTTPException, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import joblib
import pandas as pd
import logging
import time
import json

from google.cloud.logging_v2.handlers import CloudLoggingHandler
from google.cloud import logging as gcp_logging

# --- OpenTelemetry Setup ---
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.exporter.cloud_monitoring import CloudMonitoringMetricsExporter
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

# Initialize Tracer
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)
span_processor = BatchSpanProcessor(CloudTraceSpanExporter())
trace.get_tracer_provider().add_span_processor(span_processor)

# Initialize Cloud Logging
client = gcp_logging.Client()
cloud_handler = CloudLoggingHandler(client)
cloud_handler.setLevel(logging.INFO)

logger = logging.getLogger("iris-fastapi-logger")
logger.setLevel(logging.INFO)
logger.addHandler(cloud_handler)

# Initialize Cloud Monitoring Metrics
exporter = CloudMonitoringMetricsExporter()
reader = PeriodicExportingMetricReader(exporter)
provider = MeterProvider(metric_readers=[reader])
metrics.set_meter_provider(provider)
meter = metrics.get_meter(__name__)

latency_metric = meter.create_histogram(
    name="iris_request_latency_ms",
    description="Request latency for iris predictions",
    unit="ms"
)

# --- FastAPI App ---
app = FastAPI(title="🌸 Iris Classifier API")
model = joblib.load("models/model.pkl")

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

app_state = {"is_ready": False, "is_alive": True}

@app.on_event("startup")
async def startup_event():
    time.sleep(2)
    app_state["is_ready"] = True

@app.get("/live_check", tags=["Probe"])
async def liveness_probe():
    if app_state["is_alive"]:
        return {"status": "alive"}
    return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

@app.get("/ready_check", tags=["Probe"])
async def readiness_probe():
    if app_state["is_ready"]:
        return {"status": "ready"}
    return Response(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = round((time.time() - start_time) * 1000, 2)
    response.headers["X-Process-Time-ms"] = str(duration)
    return response

@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception):
    span = trace.get_current_span()
    trace_id = format(span.get_span_context().trace_id, "032x")
    logger.exception(json.dumps({
        "event": "unhandled_exception",
        "trace_id": trace_id,
        "path": str(request.url),
        "error": str(exc)
    }))
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error", "trace_id": trace_id},
    )

@app.post("/predict/")
async def predict_species(data: IrisInput, request: Request):
    with tracer.start_as_current_span("model_inference") as span:
        start_time = time.time()
        trace_id = format(span.get_span_context().trace_id, "032x")
        try:
            input_df = pd.DataFrame([data.dict()])
            prediction = model.predict(input_df)[0]
            latency = round((time.time() - start_time) * 1000, 2)
            latency_metric.record(latency)

            logger.info(json.dumps({
                "event": "prediction",
                "trace_id": trace_id,
                "input": data.dict(),
                "result": prediction,
                "latency_ms": latency,
                "status": "success"
            }))
            return {"predicted_class": prediction}
        except Exception as e:
            logger.exception(json.dumps({
                "event": "prediction_error",
                "trace_id": trace_id,
                "error": str(e)
            }))
            raise HTTPException(status_code=500, detail="Prediction failed")
