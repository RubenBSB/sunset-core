"""Google Cloud Monitoring service for tracking LLM token usage."""

import logging
import time

try:
    from google.api import label_pb2, metric_pb2
    from google.cloud import monitoring_v3

    _MONITORING_AVAILABLE = True
except ImportError:
    _MONITORING_AVAILABLE = False

logger = logging.getLogger(__name__)

METRIC_TYPE = "custom.googleapis.com/llm/token_usage"


class MonitoringService:
    """Pushes LLM token usage metrics to Google Cloud Monitoring.

    Fire-and-forget: errors are logged, never raised.
    If google-cloud-monitoring is not installed, all operations are no-ops.
    """

    def __init__(self, project: str):
        self.project = project
        self.project_name = f"projects/{project}"
        self._client = None
        self._descriptor_ensured = False
        if not _MONITORING_AVAILABLE:
            logger.warning(
                "google-cloud-monitoring not installed. "
                "Token metrics will not be pushed. "
                "Install with: pip install google-cloud-monitoring"
            )

    @property
    def client(self):
        if not _MONITORING_AVAILABLE:
            return None
        if self._client is None:
            self._client = monitoring_v3.MetricServiceClient()
        return self._client

    def _ensure_descriptor(self):
        """Create the custom metric descriptor if it doesn't exist yet."""
        if self._descriptor_ensured or not _MONITORING_AVAILABLE:
            return

        try:
            descriptor = metric_pb2.MetricDescriptor()
            descriptor.type = METRIC_TYPE
            descriptor.metric_kind = metric_pb2.MetricDescriptor.MetricKind.GAUGE
            descriptor.value_type = metric_pb2.MetricDescriptor.ValueType.INT64
            descriptor.description = "LLM token usage per request"

            for key in ("model", "method", "token_type", "tag"):
                label = label_pb2.LabelDescriptor()
                label.key = key
                label.value_type = label_pb2.LabelDescriptor.ValueType.STRING
                descriptor.labels.append(label)

            self.client.create_metric_descriptor(
                name=self.project_name, metric_descriptor=descriptor
            )
            self._descriptor_ensured = True
        except Exception as e:
            # Already exists or permission issue — either way, proceed
            if "ALREADY_EXISTS" in str(e):
                self._descriptor_ensured = True
            else:
                logger.warning(f"Failed to ensure metric descriptor: {e}")

    def write_token_metric(
        self,
        model: str,
        method: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        thinking_tokens: int = 0,
        cached_tokens: int = 0,
        tag: str = "",
    ):
        """Write token usage metrics to Cloud Monitoring.

        Writes time series points for each token type in one call.
        """
        if not _MONITORING_AVAILABLE:
            return

        try:
            self._ensure_descriptor()
            now = time.time()
            seconds = int(now)
            nanos = int((now - seconds) * 1e9)

            interval = monitoring_v3.TimeInterval(
                {"end_time": {"seconds": seconds, "nanos": nanos}}
            )

            token_types = [
                ("prompt", prompt_tokens),
                ("completion", completion_tokens),
                ("total", total_tokens),
                ("thinking", thinking_tokens),
                ("cached", cached_tokens),
            ]

            series_list = []
            for token_type, value in token_types:
                if value == 0 and token_type in ("thinking", "cached"):
                    continue
                series = monitoring_v3.TimeSeries()
                series.metric.type = METRIC_TYPE
                series.metric.labels["model"] = model
                series.metric.labels["method"] = method
                series.metric.labels["token_type"] = token_type
                series.metric.labels["tag"] = tag
                series.resource.type = "global"
                series.resource.labels["project_id"] = self.project

                point = monitoring_v3.Point(
                    {"interval": interval, "value": {"int64_value": value}}
                )
                series.points = [point]
                series_list.append(series)

            self.client.create_time_series(
                name=self.project_name, time_series=series_list
            )
            logger.debug(
                f"Wrote token metrics: model={model} method={method} tag={tag} "
                f"prompt={prompt_tokens} completion={completion_tokens} "
                f"total={total_tokens} thinking={thinking_tokens} cached={cached_tokens}"
            )
        except Exception as e:
            logger.warning(f"Failed to write token metrics: {e}")
