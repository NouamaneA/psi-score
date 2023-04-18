from typing import Union
from scipy import sparse
import numpy as np
from scipy.sparse.linalg import norm
from time import time
from progressbar import progressbar
from psi_score.utils import l_plus_m, dict_to_sparse_matrix, X

from psi_score.linalg.push import push_nf_fifo, push_fifo

def propagation_matrix(adjacency, ls, ms, lpms, solver):
    N = len(adjacency)
    A = dict()
    if solver == 'push_nf':
        A_t = {i: {} for i in adjacency}
        B_t = {i: {} for i in adjacency}
    else:
        B = {i: {} for i in range(N)}
        Deg = N*[0]
    c = []
    d = []
    for j in adjacency:
        A[j] = dict()
        for k in adjacency[j]:
            A[j][k] = ms[k] / lpms[j]
            if solver == 'push_nf':
                if ms[k] != 0:
                    A_t[k][j] = ms[k] / lpms[j]
                if ls[k] != 0:
                    B_t[k][j] = ls[k] / lpms[j]
            else:
                B[j][k] = ls[k] / lpms[j]
                Deg[j] += 1
        if ls[j]+ms[j] == 0:
            c.append(0)
            d.append(0)
        else:
            c.append(ms[j]/(ls[j]+ms[j]))
            d.append(ls[j]/(ls[j]+ms[j]))

    if solver == 'push_psi':
        # A_t = dict_to_sparse_matrix(A_t, shape=(N, N))
        return A, B, c, d

    A = dict_to_sparse_matrix(A, shape=(N, N))
    if solver == 'push_nf':
        return A_t, B_t, c, d, A
    else:
        B = dict_to_sparse_matrix(B, shape=(N, N))
        c = np.array(c)
        d = np.array(d)
        Deg = np.array(Deg)
        return A, B, c, d, Deg

def power_newsfeed(A, b_i, Deg, max_iter=500, tol=1e-4):
    p_i = b_i.copy()
    n_mult = 0
    n_msg = 0
    n_iter = 0
    for _ in range(max_iter):
        p_i_old = p_i.copy()
        nnz_idx = np.where(p_i != 0)[0]
        n_msg += Deg[nnz_idx].sum()     
        p_i = A.dot(p_i) + b_i
        n_mult += 1
        gap = np.sum(abs(p_i - p_i_old))
        # gap = norm(p_i - p_i_old, ord=1)
        n_iter += 1
        if gap < tol:
            return p_i, n_msg, n_mult, n_iter
    raise RuntimeError(f'Power-NF error: failed to converge in {max_iter} iterations.')

def wall_steady_state(i, c, p_i, d):
    if isinstance(p_i, dict):
        q_i = dict()
        for j in p_i:
            q_i[j] = c[j] * p_i[j]
            if j == i:
                q_i[i] += d[i]
        if i not in p_i:
            q_i[i] = d[i]
    else:
        N = len(d)
        d_i = d[i]*X(i, N)
        q_i = c * p_i + d_i
    return q_i

def get_psi_score(
        adjacency: dict[list], ls: Union[list, np.ndarray], 
        ms: Union[list, np.ndarray], max_iter: int =500, 
        tol: float =1e-4, solver: str ='power_psi',
        ps: list[int] =[], qs: list[int] =[] 
    ) -> tuple:
    """ Solves the Psi-score problem with a chosen algorithm.

    Parameters
    ----------
    adjacency: dict[list]
        Adjacency list of the graph where edges go from followers to leaders.
    ls: Union[list, np.ndarray]
        Posting activity of each node.
    ms: Union[list, np.ndarray]
        Re-posting activity of each node.
    solver: str
        * ``'power_psi'``, power iterations for the Psi_score vector.
        * ``'power_nf'``, for each ``i`` it uses power iterations for the vector ``p_i``, the expected probabilities to find a post of origin ``i`` on other users' NewsFeeds.
        * ``'scipy'``, use the linear system solver from the scipy.sparse library.
        * ``'push_nf'``, use push-based method for each vector ``p_i``.
        * ``'push_psi'`` use push-based method for the Psi_score vector.

    max_iter: int, optional
        Maximum number of iterations for Power-Psi and Power-NF, default=500
        
    tol: float, optional
        Tolerance for the convergence of the algorithms (except for scipy's solver), default=1e-4
    ps: list
        List of nodes ``i`` for which we want to have the ``p_i`` with the push and power_nf methods
    qs: list
        List of nodes ``i`` for which we want to have the ``q_i`` with the push and power_nf methods

    Returns
    -------
    Psi: np.ndarray
        Psi-score vector
    t: float
        Computation time in seconds.
    n_msg: int or None
        Number of messages (or update in the Psi-score vector), ``None`` for the scipy solver.
    n_mult: int or None
        Number of matrix-vector multiplications to reach convergence, ``None`` for the scipy solver.
    n_iter: int or None
        Number of iterations to reach convergence.
    P: dict[np.ndarray] (with ``solver='power_nf'``) or dict[dict] (with ``solver='push_nf'``)
        The ``p_i`` vectors of some chosen ``i`` obtained with the push_nf or the power_nf method
    Q: dict[np.ndarray] (with ``solver='power_nf'``) or dict[dict] (with ``solver='push_nf'``)
        The ``q_i`` vectors of some chosen ``i`` obtained with the push_nf or the power_nf method
    
    References
    ----------
    * Giovanidis, A., Baynat, B., Magnien, C., & Vendeville, A. (2021). 
      Ranking Online Social Users by Their Influence. IEEE/ACM Transactions on Networking, 29(5), 2198-2214. 
      https://doi.org/10.1109/tnet.2021.3085201
    * Arhachoui, N., Bautista, E., Danisch, M., & Giovanidis, A. (2022). 
      A Fast Algorithm for Ranking Users by their Influence in Online Social Platforms. 
      2022 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining (ASONAM), 526-533. 
      https://doi.org/10.1109/ASONAM55673.2022.10068673
    """

    N = len(adjacency)
    lpms = l_plus_m(adjacency, ls, ms)
    if solver == 'push_nf':
        A_t, B_t, c, d, A = propagation_matrix(adjacency, ls, ms, lpms, solver='push_nf')
    elif solver == 'push_psi':
        A, B, c, d = propagation_matrix(adjacency, ls, ms, lpms, solver='push_psi')
    else:
        A, B, c, d, Deg = propagation_matrix(adjacency, ls, ms, lpms, solver)
    t = time()
    n_mult = 0
    n_msg = 0
    n_iter = 0

    if solver == 'scipy':
        s = sparse.linalg.spsolve((sparse.identity(N)-A.T).tocsc(), c)
        Psi = 1/N * (B.T.dot(s) + d)
        t = time() - t
        return Psi, t

    elif solver == 'power_nf':
        t = time()
        Psi = []
        P = {}
        Q = {}
        for i in progressbar(range(N)):
            b_i = B.dot(X(i, N))
            p_i, n_msg_i, n_mult_i, n_iter_i = power_newsfeed(A, b_i, Deg, max_iter=max_iter, tol=tol)
            n_mult += n_mult_i
            n_msg += n_msg_i
            n_iter += n_iter_i
            q_i = wall_steady_state(i, c, p_i, d)
            Psi.append(1/N * np.sum(q_i))
            if i in ps:
                P[i] = p_i
            if i in qs:
                Q[i] = q_i
        t = time() - t
        return Psi, t, n_msg, n_mult, n_iter, P, Q

    elif solver == 'power_psi':
        At = A.T 
        s = c
        B_norm = norm(B, ord=1)
        for i in progressbar(range(max_iter)):
            s_old = s.copy()
            nnz_idx = np.where(s != 0)[0]
            n_msg += Deg[nnz_idx].sum()
            s = At.dot(s) + c
            n_mult += 1
            gap = sum(abs(s - s_old))
            gap = gap * B_norm
            n_iter += 1
            if gap < tol:
                Psi = 1/N * (B.T.dot(s) + d)
                n_mult += 1
                t = time() - t
                return Psi, t, n_msg, n_mult, n_iter
        if gap >= tol:
            raise RuntimeError(f'Power-Psi error: failed to converge in {max_iter} iterations.')

    elif solver == 'push_nf':
        Psi = []
        P = {}
        Q = {}
        tol = tol * (1 - np.max(A.sum(axis=1)))
        for i in progressbar(range(N)):
            if i in B_t:
                p_i, n_msg_i, n_iter_i = push_nf_fifo(i, A_t, B_t[i], eps=tol)
                n_msg += n_msg_i
                n_iter += n_iter_i
            else:
                p_i = dict()
            q_i = wall_steady_state(i, c, p_i, d)
            Psi.append(1/N * sum(q_i.values()))
            if i in ps:
                P[i] = p_i
            if i in qs:
                Q[i] = q_i
        t = time() - t
        Psi = np.array(Psi)
        return Psi, t, n_msg, n_iter, P, Q
    
    elif solver == 'push_psi':
        c = {i: c[i] for i in range(N) if c[i] != 0}
        d = np.array(d)
        B = dict_to_sparse_matrix(B, shape=(N, N))
        A_t = dict_to_sparse_matrix(A, shape=(N, N))
        tol = tol * (1 - np.max(A_t.sum(axis=1)))
        s, n_msg, n_iter = push_fifo(A, c, eps=tol)
        s = [s[i] if i in s else 0 for i in range(N)]
        Psi = 1/N * (B.T.dot(s) + d)

        t = time() - t

        return Psi, t, n_msg, n_iter

    
    else:
        raise ValueError('Unknown solver.')

