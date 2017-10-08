import pytest
import numpy as np
import random

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
        [0,1],
        123
    ),
])
def test_nrsampling(G,size,expected,seed):
    random.seed(seed)
    assert nrsampling(G,size) == expected


# A good test is sampling a random graph for many times and check
# if the distribution of returned subgraphs is uniform.
