from __future__ import annotations


def decide_num_ctx(loaded_context_size: int | None, required_window: int) -> int | None:
    """
    Decide which `num_ctx` value to send on an Ollama chat request.

    Ollama allocates its KV cache at model-load time, so sending a `num_ctx`
    different from what is already loaded forces a model reload. A smaller
    window never needs to force one: running with a larger loaded window than
    required still satisfies the caller.

    :param loaded_context_size: The context size the model is currently loaded with, or None when unknown.
    :param required_window: The context window this call requires.
    :returns: The `num_ctx` to send, or None when no override is needed.
    """
    if loaded_context_size is None:
        return required_window
    if loaded_context_size >= required_window:
        return None
    return required_window
