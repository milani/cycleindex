import numpy as np

def clean_matrix(A):
    """
    Removes all the vertices of graph which do not sustain any cycle
    by iteratively removing isolated vertices, sinks and sources until
    the matrix is invariant under such removal.

    Parameters
    ----------
    A : numpy.ndarray
        Adjacency matrix

    Returns
    -------
    numpy.ndarray
        The shape of this array is different from the input array
    """
    oldshape = (0,0)
    while oldshape != A.shape:
        oldshape = A.shape
        x = np.any(A,axis=0) != True
        A = np.delete(A,np.where(x)[0],1)
        A = np.delete(A,np.where(x)[0],0)
        x = np.any(A,axis=1) != True
        A = np.delete(A,np.where(x)[0],0)
        A = np.delete(A,np.where(x)[0],1)
    return A

def is_symmetric(A):
    """
    Checks if matrix A is symmetric

    Parameters
    ----------
    A : numpy.ndarray
        Adjacency matrix

    Returns
    -------
    bool
    """

    return sum(A.shape) % 2 == 0 and  np.allclose(A.transpose(1, 0), A)

def dfs(G,root = 0,seen = []):
    for i in np.nonzero(G[root,:])[0]:
        if i not in seen:
            seen.append(i)
            dfs(G,i,seen)

def is_weakly_connected(G):
    seen = []
    #Undirected structural version of G
    G = (G != 0) | (G.transpose(1,0) != 0)
    dfs(G,root=0,seen=seen)
    if len(seen) == len(G):
        return True
    return False

def calc_ratio(plus_minus,plus_plus):
    plus_minus = np.array(plus_minus)
    plus_plus = np.array(plus_plus)

    if plus_minus.ndim == 1:
        plus_minus = np.expand_dims(plus_minus,0)
        plus_plus = np.expand_dims(plus_plus,0)
    elif plus_minus.ndim == 3:
        plus_minus = np.squeeze(plus_minus,axis=0)
        plus_plus = np.squeeze(plus_plus,axis=0)

    plus_minus = np.mean(plus_minus,axis=0)
    plus_plus = np.mean(plus_plus,axis=0)
    pos = (plus_plus + plus_minus)/2
    neg = plus_plus - pos
    # avoid nan and inf
    plus_plus[np.isclose(plus_plus,0)] = 1
    ratio = neg/plus_plus
    return ratio
