import random
from typing import Any


def resolve_seed(seed: Any) -> int:
    if seed is None or str(seed).strip() == "":
        return random.SystemRandom().randint(0, 2**63 - 1)
    try:
        resolved = int(str(seed), 10)
    except ValueError as error:
        raise ValueError("seed must be an integer") from error
    if not 0 <= resolved <= 2**63 - 1:
        raise ValueError("seed must be between 0 and 2^63-1")
    return resolved
