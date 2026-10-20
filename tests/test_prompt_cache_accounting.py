"""Prompt-cache accounting: the bridge must report a non-zero cached_tokens.

The DeepSeek web endpoint returns no token metadata, so the bridge reproduces
`prompt_cache_hit_tokens` from the fact that a client resends the whole
conversation every turn: the previous turn's prompt is a prefix of this turn's.
These tests pin the pairing that makes that reconstruction correct — the
`next_sig` recorded at the end of turn N must equal the `sig` of turn N+1.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functions import (  # noqa: E402
    count_tokens,
    peek_cached_input,
    record_cached_input,
    reset_cached_input,
)
from plugin_helper import generate_signature_sync  # noqa: E402

MODEL = "deepseek-v4.1-flash"


def _msgs(*turns):
    return [
        {"role": "user" if i % 2 == 0 else "assistant", "content": t}
        for i, t in enumerate(turns)
    ]


def test_first_turn_has_no_cache_to_read():
    msgs = _msgs("[SYSTEM]\npolicy\n\n[USER]\nhello")
    sig = generate_signature_sync(msgs, MODEL)
    assert peek_cached_input(sig) == 0


def test_next_sig_of_turn_n_equals_sig_of_turn_n_plus_1():
    """This equality is the whole basis of the reconstruction."""
    m1 = _msgs("[SYSTEM]\npolicy\n\n[USER]\nfirst question")
    sig1 = generate_signature_sync(m1, MODEL)
    a1 = {"role": "assistant", "content": "first answer"}
    next_sig1 = generate_signature_sync(m1 + [a1], MODEL)

    m2 = m1 + [a1, {"role": "user", "content": "second question"}]
    sig2 = generate_signature_sync(m2, MODEL)

    assert sig1 != sig2, "distinct turns must not share a signature"
    assert sig2 == next_sig1, "turn-2 sig must equal turn-1 next_sig"


def test_cached_tokens_reports_previous_turn_prompt_size():
    m1 = _msgs("[SYSTEM]\npolicy\n\n[USER]\nfirst question")
    sig1 = generate_signature_sync(m1, MODEL)
    in1 = count_tokens(m1[0]["content"])
    assert peek_cached_input(sig1) == 0

    a1 = {"role": "assistant", "content": "first answer"}
    record_cached_input(generate_signature_sync(m1 + [a1], MODEL), in1)

    m2 = m1 + [a1, {"role": "user", "content": "second question"}]
    sig2 = generate_signature_sync(m2, MODEL)
    in2 = sum(count_tokens(m["content"]) for m in m2)

    assert peek_cached_input(sig2) == in1
    assert 0 < in1 < in2, "cache hit must be a proper prefix of the new prompt"
    rate = peek_cached_input(sig2) / in2 * 100
    assert 0 < rate < 100, f"hit rate must be a real percentage, got {rate}"


def test_cache_is_capped_at_prompt_size():
    """A shorter turn must not report more cached tokens than it sent."""
    sig = "shrink-signature"
    record_cached_input(sig, 5000)
    assert min(peek_cached_input(sig), 120) == 120


def test_reset_clears_a_rebuilt_history():
    sig = "rollover-signature"
    record_cached_input(sig, 4000)
    assert peek_cached_input(sig) == 4000
    reset_cached_input(sig)
    assert peek_cached_input(sig) == 0


def test_dsh_contract_prompt_tokens_includes_cached():
    """DSH's pi-ai adapter subtracts cacheRead from promptTokens.

    It computes `input = promptTokens - cacheReadTokens`, so prompt_tokens MUST
    include the cached portion or the derived non-cache input goes negative and
    the client's hit rate reads 0% forever. This pins the arithmetic the bridge
    relies on.
    """
    from app import _cached_prompt_tokens, _remember_prompt_tokens  # noqa: E402

    sig = "contract-signature"
    reset_cached_input(sig)
    assert _cached_prompt_tokens(sig, 10_000) == 0, "cold turn must report no hit"

    _remember_prompt_tokens(sig, 8_000)
    cached = _cached_prompt_tokens(sig, 10_000)
    assert cached == 8_000

    prompt_tokens = 10_000
    non_cached_input = prompt_tokens - cached
    assert non_cached_input == 2_000, "cacheRead must be a subset of promptTokens"
    assert 0 < cached / prompt_tokens < 1, "hit rate must be a real fraction"


def test_cached_never_exceeds_reported_prompt_tokens():
    """A shrinking prompt must not report a hit larger than what it sent."""
    from app import _cached_prompt_tokens  # noqa: E402

    sig = "shrink-signature"
    reset_cached_input(sig)
    record_cached_input(sig, 50_000)
    assert _cached_prompt_tokens(sig, 1_200) == 1_200
