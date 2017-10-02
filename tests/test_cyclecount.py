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

@pytest.mark.parametrize("A,expected", [
    (
        np.array([[0,0.5,0],[0,0,-0.5],[0.5,0,0]]),
        [0,0,-0.12500,0,0,0,0]
    ),
    (
        np.array([[0,0.5,0.5,0],[0.5,0,0.5,0],[0.5,0.5,0,0.4],[0,0,0.4,0]]),
        [0,0.91,0.25,0,0,0,0]
    ),
    (
        np.array([[0, 0.5, 0, 0, 0, 0, 0],
                  [0.5, 0, 0.4, 0.4, 0, 0, 0],
                  [0, 0.4, 0, -0.5, 0.1, 0, 0],
                  [0, 0.4, -0.5, 0, 0, 0.6, 0],
                  [0, 0, 0.1, 0, 0, 0, 0.8],
                  [0, 0, 0, 0.6, 0, 0, 0.7],
                  [0, 0, 0, 0, 0.8, 0.7, 0]]),
        [0,2.3200,-0.1600,0,-0.0336,0.010752,0]
    )
])
def test_cycle_count(A,expected):
    assert np.allclose(cycle_count(A,7),expected)
