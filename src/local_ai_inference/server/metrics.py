"""Operations: Prometheus metrics, one registry per server instance."""

import psutil
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)


class Metrics:
    content_type = CONTENT_TYPE_LATEST

    def __init__(self) -> None:
        r = self.registry = CollectorRegistry()
        self.requests = Counter(
            "lai_requests", "Finished requests by finish reason", ["finish_reason"], registry=r
        )
        self.rejected = Counter(
            "lai_rejected_requests", "Requests refused at admission", ["reason"], registry=r
        )
        self.ttft = Histogram(
            "lai_time_to_first_token_seconds",
            "Admission to first generated token",
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30),
            registry=r,
        )
        self.generated_tokens = Counter("lai_generated_tokens", "Tokens generated", registry=r)
        self.decode_tps = Gauge(
            "lai_decode_tokens_per_second", "Decode speed of the last request", registry=r
        )
        self.queue_depth = Gauge("lai_queue_depth", "Requests waiting for the slot", registry=r)
        self.context_tokens = Gauge(
            "lai_context_tokens", "Prompt + generated tokens in the active request", registry=r
        )
        self.model_load = Gauge(
            "lai_model_load_seconds", "Time to load and warm the model", registry=r
        )
        self.resident_bytes = Gauge(
            "lai_process_resident_bytes", "Memory this server process holds in RAM", registry=r
        )
        self._process = psutil.Process()

    def render(self) -> bytes:
        self.resident_bytes.set(self._process.memory_info().rss)
        return generate_latest(self.registry)
