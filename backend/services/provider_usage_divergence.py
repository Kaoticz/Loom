from enum import Enum


class ProviderUsageDivergence(str, Enum):
    """
    How the provider's prompt-token count compares with Loom's estimate.
    """

    unavailable = 'unavailable'
    agreement = 'agreement'
    input_truncated = 'input_truncated'
    estimate_too_low = 'estimate_too_low'
