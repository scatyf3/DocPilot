"""Per-model token prices (USD per 1M tokens) used to put a dollar figure on every trace."""

import logging

log = logging.getLogger(__name__)

PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    # model: (input, output)
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
    "fake": (0.0, 0.0),
}

_warned: set[str] = set()


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    prices = PRICES_PER_MTOK.get(model)
    if prices is None:
        if model not in _warned:
            log.warning("no price for model %r; recording cost as 0", model)
            _warned.add(model)
        return 0.0
    price_in, price_out = prices
    return round((input_tokens * price_in + output_tokens * price_out) / 1_000_000, 6)
