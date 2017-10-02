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
Use `cycle_count` function to count for simple cycles in the graph. It
takes a graph (numpy.ndarray) and the maximum cycle length (an integer) as the input.
It returns an array with cycle index of the graph for each cycle length upto the length provided.
```
from cycleindex import cycle_count

A = np.array([[0,0.5,0.5,0],[0.5,0,0.5,0],[0.5,0.5,0,0.4],[0,0,0.4,0]])
print(cycle_count(A,5))
```

The output is:
```
[  0.00000000e+00   9.10000000e-01   2.50000000e-01   1.37043155e-16
   0.00000000e+00]
```
Considering the triangle index, we have two triangles (0120,0210) with the weight 2*(0.5*0.5*0.5) = 0.25.

### TODO
* Implement Monte Carlo approach to evaluate the index for large networks
