import pytest
import numpy as np

from cycleindex.utils import (
    clean_matrix,
    is_symmetric
)

@pytest.mark.parametrize("A,out", [
    (
        np.array([[1,0,0],[0,0,0],[0,1,0]]),
        np.array([[1]])
    ),
    (
        np.array([[1,0,0],[0,1,0],[0,0,1]]),
        np.array([[1,0,0],[0,1,0],[0,0,1]])
    ),
    (
        np.array([[0,0.5,0,0],[0,0,0.5,0.4],[0,-0.1,0,0.2],[0,0.2,0.2,0]]),
        np.array([[0,0.5,0.4],[-0.1,0,0.2],[0.2,0.2,0]])
    )
])
def test_clean_matrix(A,out):
    assert not np.any(clean_matrix(A) != out)

@pytest.mark.parametrize("A,expected", [
    (
        np.array([[0,0.5,0],[0,0,0.5],[0.4,0,0]]),
        False
    ),
    (
        np.array([[0,0.5,0],[0.5,0,0.5],[0,0.5,0]]),
        True
    )
])
def test_is_symmetric(A,expected):
    assert is_symmetric(A) == expected


