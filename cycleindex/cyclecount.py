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
        A = np.delete(A,np.where(np.any(A,axis=0) != True )[0],1)
        A = np.delete(A,np.where(np.any(A,axis=1) != True )[0],0)
    return A

def prime_count(A,L0,Subgraph,NeighboursNumber,Primes):
    """
    Calculates the contribution to the combinatorial sieve of a
    given subgraph. This function is an implementation of the
    Eq. (2), extracting prime numbers from connected induced subgraphs

    Parameters
    ----------
    A : numpy.ndarray
        Adjacency matrix of the graph, preferably sparse 
    L0: int
        Maximum subgraph size
    Subgraph: list
        Current subgraph, a list of vertices, further vertices are
        added to this list
    Primes: list
        List regrouping the contributions of all subgraphs considered earlier

    Returns
    -------
    list
        List with the contributions of all subgraphs so far and
        now including the contribution of the subgraph
        passed to this function
    """
    SubgraphSize = len(Subgraph)
    x = A[np.meshgrid(Subgraph,Subgraph,indexing='ij')]
    xeig = np.linalg.eigvals(x);
    xS = np.power(xeig,SubgraphSize)
    mk = min(L0 , NeighboursNumber + SubgraphSize)
    
    BinomialCoeff = 1
    for k in range(SubgraphSize,mk):
        Primes[k-1] = Primes[k-1] + (-1)**k * BinomialCoeff * (-1)**SubgraphSize * sum(xS) /k
        xS = xS*xeig
        BinomialCoeff = BinomialCoeff * (SubgraphSize - k + NeighboursNumber) / (1 - SubgraphSize + k)
    Primes[mk-1] = Primes[mk-1] + (-1)**mk * BinomialCoeff * (-1)**SubgraphSize * sum(xS) / mk
    return Primes

def recursive_subgraphs(A,Anw,L0,Subgraph,AllowedVert,Primes,Neighbourhood):
    """
    Finds all the connected induced subgraphs of size up
    to "L0" of a graph known through its adjacency matrix
    "A" and containing the subgraph "Subgraph"

    Parameters
    ----------
    A : numpy.ndarray
        Adjacency matrix
    Anw: numpy.ndarray
        Undirected unweighted equivalence of A
    L0: int
        Maximum subgraph size
    Subgraph: list
        List of vertices that form current subgraph
    AllowedVert: list
        Indicator vector of pruned vertices that may be
        considered for addition to the current subgraph to
        form a larger one
    Neighbourhood: list
        Indicator vector of the vertices that are contained
        in the current subgraph or reachable via one edge

    Returns
    -------
    list
        List regrouping the contribution of all the subgraphs found so far
    """
    L = len(Subgraph)
    NeighboursNumber = len(np.nonzero(Neighbourhood)[0]) - L
    Primes = prime_count(A,L0,Subgraph,NeighboursNumber,Primes)
    if L == L0:
        return Primes
    
    Neighbours = np.where(np.array(Neighbourhood) & np.array(AllowedVert))[0]
    for j in range(len(Neighbours)):
        v = Neighbours[j]
        if len(Subgraph) > L:
            Subgraph[L] = v
        else:
            Subgraph.append(v)
        AllowedVert[v] = False
        newNeighbourhood = Neighbourhood + Anw[v,:]
        Primes = recursive_subgraphs(A,Anw,L0,Subgraph[:],AllowedVert[:],Primes,newNeighbourhood[:])
    
    return Primes
    
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

    return np.allclose(A.transpose(1, 0), A)

def cycle_count(A,L0):
    """
    Counts all simple cycles of length up to L0 included on a
    graph whose adjacency matrix is A.

    Parameters
    ----------
    A : numpy.ndarray
        Adjacency matrix
    L0: int
        Length of cycles to count

    Returns
    -------
    list
        Index "i" is the number of primes of length i>=1 up to L0
    """
    Primes = np.zeros(L0)
    np.fill_diagonal(A,0)
    A = clean_matrix(A)

    if is_symmetric(A):
        Anw = A != 0
    else:
        Anw = ( A != 0 ) | ( A.transpose(1,0) != 0)

    Size = len(A)
    
    if L0 > Size:
        L0 = Size
    
    AllowedVert = [True]*Size
    for i in range(len(A)):
        AllowedVert[i] = False
        Neighbourhood = [False]*Size
        Neighbourhood[i] = True
        Neighbourhood = Neighbourhood + Anw[i,:]
        Primes = recursive_subgraphs(A,Anw,L0,[i],AllowedVert[:],Primes,Neighbourhood[:])
    
    return Primes

__all__ = ['cycle_count']
