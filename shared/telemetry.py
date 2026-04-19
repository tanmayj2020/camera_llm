"""OpenTelemetry instrumentation for VisionBrain.

Provides distributed tracing, metrics, and structured logging.
Configured via environment variables:
  - OTEL_EXPORTER_OTLP_ENDPOINT: gRPC endpoint for traces/metrics
  - OTEL_SERVICE_NAME: service name (default: visionbrain)
"""

import logging
import os
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable

logger = logging.getLogger(__name__)

_tracer = None
_meter = None
_initialized = False


def init_telemetry(service_name: str | None = None):
    """Initialize OpenTelemetry SDK. Call once at application startup."""
    global _tracer, _meter, _initialized

    if _initialized:
        return

    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "")
    svc = service_name or os.environ.get("OTEL_SERVICE_NAME", "visionbrain")

    if not endpoint:
        logger.info("OTEL_EXPORTER_OTLP_ENDPOINT not set — telemetry disabled (noop)")
        _initialized = True
        return

    try:
        from opentelemetry import metrics, trace
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource.create({"service.name": svc})

        # Tracing
        tracer_provider = TracerProvider(resource=resource)
        tracer_provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=True))
        )
        trace.set_tracer_provider(tracer_provider)
        _tracer = trace.get_tracer(svc)

        # Metrics
        metric_reader = PeriodicExportingMetricReader(
            OTLPMetricExporter(endpoint=endpoint, insecure=True),
            export_interval_millis=15000,
        )
        meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
        metrics.set_meter_provider(meter_provider)
        _meter = metrics.get_meter(svc)

        _initialized = True
        logger.info("OpenTelemetry initialized: endpoint=%s service=%s", endpoint, svc)

    except ImportError:
        logger.info("opentelemetry packages not installed — telemetry disabled")
        _initialized = True
    except Exception as e:
        logger.warning("OpenTelemetry init failed: %s", e)
        _initialized = True


def get_tracer():
    """Get the global tracer (noop-safe)."""
    if _tracer is not None:
        return _tracer
    try:
        from opentelemetry import trace
        return trace.get_tracer("visionbrain")
    except ImportError:
        return _NoopTracer()


def get_meter():
    """Get the global meter (noop-safe)."""
    if _meter is not None:
        return _meter
    try:
        from opentelemetry import metrics
        return metrics.get_meter("visionbrain")
    except ImportError:
        return _NoopMeter()


def traced(name: str | None = None):
    """Decorator to trace a function."""
    def decorator(func: Callable) -> Callable:
        span_name = name or f"{func.__module__}.{func.__qualname__}"

        @wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_tracer()
            with tracer.start_as_current_span(span_name) as span:
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            tracer = get_tracer()
            with tracer.start_as_current_span(span_name) as span:
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper
    return decorator


# ---- Noop stubs when OTel is not installed ----

class _NoopSpan:
    def set_attribute(self, *a, **kw): pass
    def set_status(self, *a, **kw): pass
    def record_exception(self, *a, **kw): pass
    def __enter__(self): return self
    def __exit__(self, *a): pass

class _NoopTracer:
    def start_as_current_span(self, *a, **kw): return _NoopSpan()
    def start_span(self, *a, **kw): return _NoopSpan()

class _NoopMeter:
    def create_counter(self, *a, **kw): return _NoopInstrument()
    def create_histogram(self, *a, **kw): return _NoopInstrument()
    def create_up_down_counter(self, *a, **kw): return _NoopInstrument()

class _NoopInstrument:
    def add(self, *a, **kw): pass
    def record(self, *a, **kw): pass
