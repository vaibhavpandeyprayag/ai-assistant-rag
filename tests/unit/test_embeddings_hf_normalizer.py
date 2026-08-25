"""Unit tests for the Hugging Face embedding payload normalizer."""

import pytest

from app.embeddings.hf_api import _to_vector


def test_flat_vector_passes_through_as_floats() -> None:
    assert _to_vector([1, 2.5, -3]) == [1.0, 2.5, -3.0]


def test_token_matrix_is_mean_pooled() -> None:
    matrix = [[1.0, 3.0], [2.0, 4.0], [3.0, 5.0]]
    assert _to_vector(matrix) == [2.0, 4.0]


@pytest.mark.parametrize("payload", ["not-a-vector", b"bytes", [], [[]]])
def test_invalid_payloads_raise_value_error(payload: object) -> None:
    with pytest.raises(ValueError):
        _to_vector(payload)
