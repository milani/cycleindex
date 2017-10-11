import numpy as np
from multiprocessing import Pool, cpu_count
from cyclecount import cycle_count
from sampling import nrsampling, vxsampling
from utils import clean_matrix, calc_ratio


def batch_count_(G, length, batch_size, sampling_func=nrsampling, exact_subgraph_size=True, counts=([], [])):
    for i in range(batch_size):
        subgraph = sampling_func(G, length, exact_subgraph_size)
        count = cycle_count(G[np.ix_(subgraph, subgraph)], length)
        counts[0].append(count[0])
        counts[1].append(count[1])
    return counts


def batch_count_parallel_(G, length, batch_size, n_cores, pool, sampling_func=nrsampling, exact_subgraph_size=True, counts=([], [])):
    promises = [pool.apply_async(batch_count_, args=(G, length, batch_size / n_cores, sampling_func, exact_subgraph_size))
                for i in range(n_cores)]
    count = ([], [])
    for p in promises:
        res = p.get()
        count[0].append(res[0])
        count[1].append(res[1])
    counts[0].append(sum(count[0], []))
    counts[1].append(sum(count[1], []))

    return counts


def balance_ratio(G, length, exact=False, n_samples=100000, accuracy=None, parallel=True):
    """
    Returns balance ratio of "G" based on simple cycles of length upto "length". Please note
    that it catches the Keyboard interruption if "exact" is False. It is helpful when one wants
    to interrupt the algorithm because it is running for longer than expected or the accuracy
    has reached an acceptable value.

    Parameters
    ----------
    G : numpy.ndarray
        Adjacency matrix of graph
    length : int
        Maximum length of simple cycles
    exact : bool
        If True, the algorithm counts all subgraphs of G.
        Otherwise, it uses a sampling technique to sample
        enough number of subgraphs so that the estimate of balance
        would converge.
    n_samples : int
        If exact is False, how many samples to take from graph. Use a float number
    accuracy : float
        If provided, it is used as a threshold on standard deviation of estimated ratios. In this
        case, the "n_samples" parameter is ignored and algorithm runs until the desired accuracy
        is reached.
    parallel : bool
        If True, the sampling is done in parallel.
        If "exact" is True, "parallel" is ignored.

    Returns
    -------
    numpy.ndarray
        A numpy array containing balance ratios upto desired length "length".
    """
    if exact:
        counts = cycle_count(G, length)
    else:
        counts = ([], [])
        batch_count = batch_count_
        args = (nrsampling, True, counts)

        if parallel:
            n_cores = cpu_count()
            pool = Pool(processes=n_cores)
            batch_count = batch_count_parallel_
            args = (n_cores, pool, nrsampling, True, counts)

        if accuracy:
            last_ratios = []
            batch_size = 1000
            deviation = accuracy

            while np.any(deviation >= accuracy):
                batch_count(G, length, batch_size, *args)
                ratios = calc_ratio(*counts)
                last_ratios.append(ratios)
                if len(last_ratios) > 5:
                    deviation = np.std(last_ratios, axis=0)
        else:
            batch_count(G, length, n_samples, *args)

    return calc_ratio(*counts)

__all__ = ['clean_matrix', 'cycle_count', 'nrsampling', 'vxsampling', 'balance_ratio']
