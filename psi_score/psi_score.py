import numpy as np
from psi_score.psi_solvers import get_psi_score
from typing import Union

class PsiScore:
    """Psi-score of each node of a graph. Influence measure in Online Social Platforms.

    If ``lambdas[i]`` is the same for all ``i`` and ``mus[i]`` is the same for all ``i``, then Psi-score gives the PageRank vector.

    Parameters
    ----------
    solver: str
        * ``'power_psi'``, power iterations for the Psi_score vector.
        * ``'power_nf'``, for each ``i`` it uses power iterations for the vector ``p_i``, the expected probabilities to find a post of origin ``i`` on other users' NewsFeeds.
        * ``'scipy'``, use the linear system solver from the scipy.sparse library.

    n_iter: int, optional
        Maximum number of iterations for Power-Psi and Power-NF, default=1000

    tol: float, optional
        Tolerance for the convergence of Power-Psi and Power-NF, default=1e-6

    Attributes
    ----------
    scores: np.ndarray
        Psi-score of each node.
    t: float
        Computation time in seconds.
    n_msg: int or None
        Number of messages (or update in the Psi-score vector), ``None`` for the scipy solver.
    n_mult: int or None
        Number of matrix-vector multiplications to reach convergence, ``None`` for the scipy solver.

    Example
    -------
    >>> from psi_score import PsiScore
    >>> adjacency = {0: [1, 3], 1: [0, 2], 2: [0, 1, 3], 3: [0]}
    >>> lambdas = [0.23, 0.50, 0.86, 0.19]
    >>> mus = [0.42, 0.17, 0.10, 0.37]
    >>> psiscore = PsiScore()
    >>> scores = psiscore.fit_transform(adjacency, lambdas, mus)
    >>> scores
    array([0.21158803, 0.35253745, 0.28798439, 0.14789014])
    >>> np.round(scores, 2)
    array([0.21, 0.35, 0.29, 0.15])

    References
    ----------
    * Giovanidis, A., Baynat, B., Magnien, C., & Vendeville, A. (2021). 
      Ranking Online Social Users by Their Influence. IEEE/ACM Transactions on Networking, 29(5), 2198–2214. 
      https://doi.org/10.1109/tnet.2021.3085201
    """

    def __init__(self, solver: str = 'power_psi', n_iter: int = 1000, tol: float = 1e-6):
        super(PsiScore, self).__init__()
        self.solver = solver
        self.n_iter = n_iter
        self.tol = tol
        self.scores = None
        self.time = None
        self.n_msg = None
        self.n_mult = None

    def fit(self, adjacency: dict[list], lambdas: Union[list, np.ndarray], mus: Union[list, np.ndarray]) -> 'PsiScore':
        """Fit algorithm.
        
        Parameters
        ----------
        adjacency:
            Adjacency list of the graph where edges go from followers to leaders.
        lambdas:
            Posting activity of each node.
        mus:
            Re-posting activity of each node.
        """
        if self.solver == 'scipy':
            self.scores, self.time = get_psi_score(adjacency, lambdas, mus, n_iter=self.n_iter, solver=self.solver, tol=self.tol)
        else:
            self.scores, self.time, self.n_msg, self.n_mult = get_psi_score(adjacency, lambdas, mus, n_iter=self.n_iter, solver=self.solver, tol=self.tol)        
        return self

    def fit_transform(self, *args, **kwargs) -> np.ndarray:
        """Fit algorithm and return the Psi-score vector.

        Parameters
        ----------
        Same as the ``fit`` method.

        Returns
        -------
        scores: np.ndarray
            Psi-score vector.
        
        """
        self.fit(*args, **kwargs)
        return self.scores