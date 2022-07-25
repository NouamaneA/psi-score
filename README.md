# Psi-score

$\psi$-score: Metric of user influence in Online Social Networks

## Requirements
* Python >=3.9,<3.11

## Installation

```bash
$ pip install psi-score
```

## Usage

```python
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
```
You can use another algorithm and change some parameters:
```python
>>> psiscore = PsiScore(solver='power_nf', n_iter=500, tol=1e-3)
>>> scores = psiscore.fit_transform(adjacency, lambdas, mus, ps=[1], qs=[0, 3])
```
The ``ps`` and ``qs`` parameters allows to have some chosen ``p_i`` and ``q_i`` vectors (only with the ``push`` and ``power_nf`` methods):
```python
>>> psiscore.P
{1: array([0.5333334 , 0.1681094 , 0.46801851, 0.34442264])}
>>> psiscore.Q
{0: array([0.46164044, 0.0514935 , 0.02798624, 0.30484491]),
 3: array([0.13087053, 0.01616898, 0.01850541, 0.42554885])}
```

## License

`psi-score` was created by Nouamane Arhachoui. It is licensed under the terms of the MIT license.

## References

* Giovanidis, A., Baynat, B., Magnien, C., & Vendeville, A. (2021).
  Ranking Online Social Users by Their Influence. 
  IEEE/ACM Transactions on Networking, 29(5), 2198–2214. https://doi.org/10.1109/tnet.2021.3085201

* Arhachoui, N., Bautista, E., Danisch, M., & Giovanidis, A. (2022). 
  A Fast Algorithm for Ranking Users by their Influence in Online Social Platforms. 
  arXiv Preprint. https://doi.org/10.48550/ARXIV.2206.09960
