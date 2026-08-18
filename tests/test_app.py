import pytest

from h3_seed import resolve_seed


def test_blank_seed_resolves_to_valid_random_seed():
    first = resolve_seed("")
    second = resolve_seed(None)
    assert 0 <= first <= 2**63 - 1
    assert 0 <= second <= 2**63 - 1


@pytest.mark.parametrize("value", ["42", " 42 "])
def test_seed_accepts_integer_text(value):
    assert resolve_seed(value) == 42


@pytest.mark.parametrize("value", ["abc", "1.5", "-1", str(2**63)])
def test_seed_rejects_invalid_text(value):
    with pytest.raises(ValueError, match="seed"):
        resolve_seed(value)
