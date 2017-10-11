Cycle Index
===========

This package implements Cycle Index using the algorithm described in the paper "A general purpose algorithm for counting simple cycles and simple paths of any length"[1].

The cycle counting algorithm is a python implementation of [Matlab code](https://uk.mathworks.com/matlabcentral/fileexchange/60814-cyclecount-a-l0-) provided by the author of the paper.

[1] Pierre-Louis Giscard, Nils Kriege, Richard C. Wilson; A general purpose algorithm for counting simple cycles and simple paths of any length, https://arxiv.org/abs/1612.05531

## How to Install
```
pip install git+https://github.com/milani/cycleindex.git
```

## How to use
For a complete example please visit [examples/](examples/).

Here is a quick demonstration.

Having an undirected graph of 4 nodes like below, we want to count the cycles of upto length 5 and calculate the balance ratio for each length. Please note that the graph is weighted and each edge has weight equal to 1 except the edge `(1,2)` which has the weight of -1. 

```
1  2
*--*
+  +
*++*
4  3
```

First, calculate the cycles.

```python
>>> import numpy as np
>>> from cycleindex import balance_ratio, cycle_count, clean_matrix
>>>
>>> A = np.array([[0,-1,0,1],[-1,0,1,0],[0,1,0,1],[1,0,1,0]])
>>> cycle_count(A,5)
```

The output is:

```
([0.0, 4.0, 1.3322676295501878e-15, -2.0000000000000036, 0], [0.0, 4.0, 1.3322676295501878e-15, 1.9999999999999964, 0])
```

which is a tuple of two lists. The first list shows `N_pos - N_neg` and the second list shows `N_pos + N_neg`. Here, we have 4 backtracks, 0 triangles and 2 negative square graphs. If you are wondering why 2 negative C4, that's because the algorithm works on directed graphs so an undirected edge is equivalent to two directed ones, hence two square graphs in A.

To calculate ratio `R = -N_neg / ( N_pos + N_neg )`, we can use `balance_ratio` function:

```python
>>> balance_ratio(A,4,exact=True)
```

```
array([ 0.,  0.,  0.,  1.])
```

It shows that 100% of C4s in `A` have negative weight.


## Additional functionalities

* Works for both directed and undirected graphs.
* Works with weighted graphs.
* Has sampling capabilities for very large graphs to estimate `balance_ratio`
* Supports multiprocessing to calculates ratios even faster.
