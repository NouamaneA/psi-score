# Psi-score

Metric of user influence in Online Social Networks

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
You can use another algorithm than the default one as well as the tolerance:
```python
>>> psiscore = PsiScore(solver='power_nf', n_iter=500, tol=1e-3)
>>> scores = psiscore.fit_transform(adjacency, lambdas, mus)
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
