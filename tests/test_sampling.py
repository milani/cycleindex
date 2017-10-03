import pytest
import numpy as np

from cycleindex.sampling import (
    nrsampling
)

@pytest.mark.parametrize("G,size,expected,seed", [
    (
        np.array([[0,1,0],[0,0,0],[0,1,0]]),
        3,
        [0,1,2],
        123
    ),
    (
        np.array([[0,1,0],[0,0,0],[0,0,0]]),
        3,
        [0,1],
        123
    ),
    (
        np.array([[0,1,0],[0,0,1],[1,0,0]]),
        2,
        [1,2],
        123
    ),
])
def test_clean_matrix(G,size,expected,seed):
    np.random.seed(seed)
    assert nrsampling(G,size) == expected


# A good test is sampling a random graph for many times and check
# if the distribution of returned subgraphs is uniform.