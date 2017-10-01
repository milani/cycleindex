import pytest
import numpy as np

from cycleindex.cyclecount import (
    clean_matrix,
    is_symmetric,
    cycle_count
)

@pytest.mark.parametrize("A,out", [
    (
        np.array([[1,0,0],[0,0,0],[0,1,0]]),
        np.array([[1,0],[0,1]])
    ),
    (
        np.array([[1,0,0],[0,1,0],[0,0,1]]),
        np.array([[1,0,0],[0,1,0],[0,0,1]])
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

@pytest.mark.parametrize("A,expected", [
    (
        np.array([[0,0.5,0],[0,0,-0.5],[0.5,0,0]]),
        [0,0,-0.12500,0,0]
    ),
    (
        np.array([[0,0.5,0.5,0],[0.5,0,0.5,0],[0.5,0.5,0,0.4],[0,0,0.4,0]]),
        [0,0.91,0.25,0,0]
    )
])
def test_cycle_count(A,expected):
    print(cycle_count(A,5))
    assert np.allclose(cycle_count(A,5),expected)
