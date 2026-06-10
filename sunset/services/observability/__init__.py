"""OpenTelemetry bootstrap for sunset services.

Pushes traces and metrics over OTLP/HTTP to Grafana Cloud (or any OTLP
endpoint). Auto-instruments FastAPI, asyncpg, redis, httpx so HTTP / DB /
Redis / outbound-LLM-SDK spans are free.

Wire it once at process startup, before FastAPI/worker objects are
constructed:

    from sunset.services.observability import init_observability
    init_observability(service_name="myproject-api")
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_initialized = False


def init_observability(
    service_name: str,
    service_version: Optional[str] = None,
    environment: Optional[str] = None,
) -> None:
    global _initialized
    if _initialized:
        return

    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    headers = os.environ.get("OTEL_EXPORTER_OTLP_HEADERS", "").strip()
    # Fall back to Secret Manager so prod doesn't need Cloud Run env wiring.
    if not endpoint:
        try:
            from sunset.services.secrets import get_secrets

            secrets = get_secrets()
            endpoint = secrets.get_secret(
                "otel-exporter-otlp-endpoint", default=""
            ).strip()
            if not headers:
                headers = secrets.get_secret(
                    "otel-exporter-otlp-headers", default=""
                ).strip()
        except Exception:
            logger.warning(
                "Failed to read OTel config from Secret Manager", exc_info=True
            )

    if not endpoint:
        logger.info("OTEL disabled (no endpoint in env or Secret Manager)")
        _initialized = True
        return

    # Re-inject into the process env so the OTLP exporters pick them up via
    # their default env-based config path.
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = endpoint
    if headers:
        os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = headers

    from opentelemetry import metrics, trace
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
        OTLPMetricExporter,
    )
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter,
    )
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    resource = Resource.create(
        {
            "service.name": service_name,
            "service.version": service_version or os.environ.get("GIT_SHA", "dev"),
            "deployment.environment": environment or os.environ.get("ENV", "local"),
        }
    )

    # Traces — endpoint should be base URL, exporter appends /v1/traces.
    trace_provider = TracerProvider(resource=resource)
    trace_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(trace_provider)

    # Metrics — exporter appends /v1/metrics. 15s push cadence balances
    # freshness against egress cost on Grafana Cloud free tier.
    metric_reader = PeriodicExportingMetricReader(
        OTLPMetricExporter(),
        export_interval_millis=15_000,
    )
    meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    metrics.set_meter_provider(meter_provider)

    _install_auto_instrumentations()
    # Force metric definitions to register against the just-set provider.
    from sunset.services.observability import metrics as _metrics  # noqa: F401

    _initialized = True
    logger.info(
        "OTEL initialized: service=%s endpoint=%s env=%s",
        service_name,
        endpoint,
        environment or os.environ.get("ENV", "local"),
    )


def _install_auto_instrumentations() -> None:
    try:
        from opentelemetry.instrumentation.asyncpg import AsyncPGInstrumentor

        AsyncPGInstrumentor().instrument()
    except Exception:
        logger.warning("asyncpg instrumentation failed", exc_info=True)

    try:
        from opentelemetry.instrumentation.redis import RedisInstrumentor

        RedisInstrumentor().instrument()
    except Exception:
        logger.warning("redis instrumentation failed", exc_info=True)

    try:
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

        HTTPXClientInstrumentor().instrument()
    except Exception:
        logger.warning("httpx instrumentation failed", exc_info=True)


def instrument_fastapi(app) -> None:
    """Call after FastAPI app creation. Skipped if observability is disabled."""
    if not _initialized or not os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        return
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app)
    except Exception:
        logger.warning("FastAPI instrumentation failed", exc_info=True)
