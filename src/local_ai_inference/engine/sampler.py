"""Turn logits into the next token: greedy, or temperature + top-p (nucleus) sampling."""

import numpy as np


def sample(logits: np.ndarray, temperature: float, top_p: float, rng: np.random.Generator) -> int:
    if temperature <= 0:
        return int(np.argmax(logits))  # greedy: deterministic, used for parity tests

    z = logits.astype(np.float64) / temperature
    probs = np.exp(z - z.max())
    probs /= probs.sum()

    if top_p < 1.0:
        order = np.argsort(probs)[::-1]
        cumulative = np.cumsum(probs[order])
        keep = order[: int(np.searchsorted(cumulative, top_p)) + 1]  # smallest set reaching top_p
        kept = probs[keep] / probs[keep].sum()
        return int(rng.choice(keep, p=kept))

    return int(rng.choice(len(probs), p=probs))
