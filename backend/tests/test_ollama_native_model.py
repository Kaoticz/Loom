from services.ollama_native_model import decide_num_ctx


def test_nothing_loaded_requests_the_required_window() -> None:
    """
    Without a running instance there is nothing to preserve.
    """
    assert decide_num_ctx(None, 32_768) == 32_768


def test_a_satisfied_window_sends_no_override() -> None:
    """
    A loaded window at or above the requirement must not force a reload.
    """
    assert decide_num_ctx(32_768, 32_768) is None
    assert decide_num_ctx(65_536, 32_768) is None


def test_a_smaller_loaded_window_requests_the_larger_window() -> None:
    """
    Only a genuinely larger requirement justifies a reload.
    """
    assert decide_num_ctx(4_096, 32_768) == 32_768
