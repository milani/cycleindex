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


