"""Output control: stop strings.

When streaming, text that might be the start of a stop string is held back
until we know whether the stop string completes.
"""

from collections.abc import Iterable


class StopMatcher:
    def __init__(self, stops: Iterable[str] = ()) -> None:
        self._stops = [s for s in stops if s]
        self._hold = max((len(s) for s in self._stops), default=1) - 1
        self._buf = ""

    def push(self, text: str) -> tuple[str, bool]:
        """Add new text. Returns (text safe to emit, whether a stop string was hit)."""
        self._buf += text
        hits = [i for s in self._stops if (i := self._buf.find(s)) != -1]
        if hits:
            out, self._buf = self._buf[: min(hits)], ""
            return out, True
        if len(self._buf) <= self._hold:
            return "", False
        cut = len(self._buf) - self._hold
        out, self._buf = self._buf[:cut], self._buf[cut:]
        return out, False

    def flush(self) -> str:
        out, self._buf = self._buf, ""
        return out
