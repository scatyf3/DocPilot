import pytest

from docpilot.pricing import cost_usd


def test_opus_cost():
    # 1M in at $4 + 0.5M out at $20/M = $14
    assert cost_usd("claude-opus-5-5", 1_000_000, 500_000) == pytest.approx(14.0)


def test_small_numbers_round_to_micro_dollars():
    assert cost_usd("claude-haiku-5-5", 1234, 567) == pytest.approx(0.000407, abs=1e-6)


def test_unknown_model_is_free_with_warning(caplog):
    assert cost_usd("mystery-model", 1000, 1000) == 0.0
    assert "mystery-model" in caplog.text
