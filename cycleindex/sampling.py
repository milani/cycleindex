def nrsampling(G,size,exact=False):
    """
    Uses NRS algorithm[1] to uniformly sample subgraphs of size "size" from graph "G".

    [1] Lu X., Bressan S. (2012) Sampling Connected Induced Subgraphs Uniformly at Random.
        In: Ailamaki A., Bowers S. (eds) Scientific and Statistical Database Management.
        SSDBM 2012. Lecture Notes in Computer Science, vol 7338. Springer, Berlin, Heidelberg

    Parameters
    ----------
    G : numpy.ndarray
        Adjacency matrix of the graph. It should not contain self-loops.
    size : int
        Size of subgraph to sample
    exact : bool
        If True, the algorithm tries to find a subgraph that matches the size exactly.
        It means if the graph G does not include such subgraph, the function never returns.
        If False, the algorithm returns a subgraph that might be smaller than the required size,
        even if such subgraph exists in the graph.

    Returns
    -------
    list
        List of vertices that form the sampled subgraph.
    """
    subgraph = []
    allowed = np.array([True]*len(G))
    neighbourhood = np.array([False]*len(G))

    neighbourhood[np.random.randint(0,len(G)-1)] = True

    # random vertex expansion
    while len(subgraph) < size:
        neighbours = np.where(neighbourhood & allowed)[0]
        if len(neighbours) == 0:
            if not exact:
                return subgraph
            else:
                subgraph=[]
                allowed = np.array([True]*len(G))
                neighbourhood = np.array([False]*len(G))
                neighbourhood[np.random.randint(0,len(G)-1)] = True
                continue

        u = np.random.choice(neighbours)
        allowed[u] = False
        neighbourhood[u] = True
        neighbourhood = neighbourhood + (G[u,:] != 0) + (G[:,u] != 0) # direction is not important
        subgraph.append(u)
        old_neighbours_size = len(neighbours)

    # Fix for bias toward subgraphs with higher clustering coef.
    i = size
    neighbours = np.where(neighbourhood & allowed)[0]
    while len(neighbours) > 0:
        i += 1
        v = np.random.choice(neighbours)
        alpha = np.random.random_sample()
        if alpha < float(size)/i :
            u_index = np.random.randint(0,len(subgraph)-1)
            u = subgraph[u_index]
            
            s_prime = np.concatenate([[v],np.delete(subgraph,u_index)])
            s_prime_adj = G[np.meshgrid(s_prime,s_prime,indexing='ij')]
            # Check if undirected version of the S' is connected by checking degree of all its vertices
            if len(np.where(np.any(s_prime_adj + s_prime_adj.transpose(1,0),axis=0) == False)[0]) == 0:
                subgraph[u_index] = v
        allowed[v] = False
        neighbourhood = neighbourhood + (G[v,:] != 0) + (G[:,v] != 0)
        neighbours = np.where(neighbourhood & allowed)[0]

    return subgraph
