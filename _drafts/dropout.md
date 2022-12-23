---
title: Dropout
---

## Forward Propagation

$$
\begin{align}
a_{i, k}^{[l]} &= \text{?}, \\
r_{i, k}^{[l]} &\sim \bernoulli(p_k^{[l]}) =
\begin{cases}
1 &\text{with probability } p_k^{[l]}, \\
0 &\text{with probability } 1 - p_k^{[l]},
\end{cases} \\
\tilde{a}_{i, k}^{[l]} &= \frac{r_{i, k}^{[l]}}{p_k^{[l]}} a_{i, k}^{[l]}, \\
z_{i, j}^{[l + 1]} &= \sum_k w_{j, k}^{[l + 1]} \tilde{a}_{i, k}^{[l]} + b_{j}^{[l + 1]}, \\
a_{i, j}^{[l + 1]} &= g_j^{[l + 1]}(\dots, z_{i, j - 1}^{[l + 1]}, z_{i, j}^{[l + 1]}, z_{i, j + 1}^{[l + 1]}, \dots).
\end{align}
$$

## Backpropagation

$$
\begin{align}
\pdv{J}{a_{i, j}^{[l + 1]}} &= \text{?}, \\
\pdv{J}{z_{i, j}^{[l + 1]}} &= \sum_\jj \pdv{J}{a_{i, \jj}^{[l + 1]}} \pdv{a_{i, \jj}^{[l + 1]}}{z_{i, j}^{[l + 1]}}, \\
\pdv{J}{w_{j, k}^{[l + 1]}} &= \sum_i \pdv{J}{z_{i, j}^{[l + 1]}} \pdv{z_{i, j}^{[l + 1]}}{w_{j, k}^{[l + 1]}} = \sum_i \pdv{J}{z_{i, j}^{[l + 1]}} \tilde{a}_{i, k}^{[l]}, \\
\pdv{J}{b_j^{[l + 1]}} &= \sum_i \pdv{J}{z_{i, j}^{[l + 1]}} \pdv{z_{i, j}^{[l + 1]}}{b_j^{[l + 1]}} = \sum_i \pdv{J}{z_{i, j}^{[l + 1]}}, \\
\pdv{J}{\tilde{a}_{i, k}^{[l]}} &= \sum_j \pdv{J}{z_{i, j}^{[l + 1]}} \pdv{z_{i, j}^{[l + 1]}}{\tilde{a}_{i, k}^{[l]}} = \sum_j \pdv{J}{z_{i, j}^{[l + 1]}} w_{j, k}^{[l + 1]}, \\
\pdv{J}{a_{i, k}^{[l]}} &= \pdv{J}{\tilde{a}_{i, k}^{[l]}} \pdv{\tilde{a}_{i, k}^{[l]}}{a_{i, k}^{[l]}} = \pdv{J}{\tilde{a}_{i, k}^{[l]}} \frac{r_{i, k}^{[l]}}{p_k^{[l]}}.
\end{align}
$$
