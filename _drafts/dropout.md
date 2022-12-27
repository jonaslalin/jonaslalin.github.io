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
z_{i, j}^{[l + 1]} &= \sum_{k = 1}^{n^{[l]}} w_{j, k}^{[l + 1]} \tilde{a}_{i, k}^{[l]} + b_j^{[l + 1]}, \\
a_{i, j}^{[l + 1]} &= g_j^{[l + 1]}(z_{i, 1}^{[l + 1]}, \dots, z_{i, j}^{[l + 1]}, \dots, z_{i, n^{[l + 1]}}^{[l + 1]}).
\end{align}
$$

## Backpropagation

Assume

$$
\begin{equation}
\pdv{\J}{a_{i, j}^{[l + 1]}} = \text{?}
\end{equation}
$$

is given.

$$z_{i, j}^{[l + 1]}$$ affects $$a_{i, \jj}^{[l + 1]}$$, $$\jj = 1, \dots, n^{[l + 1]}$$, since

$$
\begin{align}
a_{i, 1}^{[l + 1]} &= g_1^{[l + 1]}(z_{i, 1}^{[l + 1]}, \dots, z_{i, j}^{[l + 1]}, \dots, z_{i, n^{[l + 1]}}^{[l + 1]}), \\
&\vdotswithin{=} \notag \\
a_{i, \jj}^{[l + 1]} &= g_\jj^{[l + 1]}(z_{i, 1}^{[l + 1]}, \dots, z_{i, j}^{[l + 1]}, \dots, z_{i, n^{[l + 1]}}^{[l + 1]}), \\
&\vdotswithin{=} \notag \\
a_{i, n^{[l + 1]}}^{[l + 1]} &= g_{n^{[l + 1]}}^{[l + 1]}(z_{i, 1}^{[l + 1]}, \dots, z_{i, j}^{[l + 1]}, \dots, z_{i, n^{[l + 1]}}^{[l + 1]});
\end{align}
$$

hence,

$$
\begin{equation}
\pdv{\J}{z_{i, j}^{[l + 1]}} = \sum_{\jj = 1}^{n^{[l + 1]}} \pdv{\J}{a_{i, \jj}^{[l + 1]}} \pdv{a_{i, \jj}^{[l + 1]}}{z_{i, j}^{[l + 1]}}.
\end{equation}
$$

Both $$w_{j, k}^{[l + 1]}$$ and $$b_j^{[l + 1]}$$ affect $$z_{i, j}^{[l + 1]}$$, $$i = 1, \dots, m$$, since

$$
\begin{align}
z_{1, j}^{[l + 1]} &= \sum_{k = 1}^{n^{[l]}} w_{j, k}^{[l + 1]} \tilde{a}_{1, k}^{[l]} + b_j^{[l + 1]}, \\
&\vdotswithin{=} \notag \\
z_{i, j}^{[l + 1]} &= \sum_{k = 1}^{n^{[l]}} w_{j, k}^{[l + 1]} \tilde{a}_{i, k}^{[l]} + b_j^{[l + 1]}, \\
&\vdotswithin{=} \notag \\
z_{m, j}^{[l + 1]} &= \sum_{k = 1}^{n^{[l]}} w_{j, k}^{[l + 1]} \tilde{a}_{m, k}^{[l]} + b_j^{[l + 1]};
\end{align}
$$

hence,

$$
\begin{align}
\pdv{\J}{w_{j, k}^{[l + 1]}} &= \sum_i \pdv{\J}{z_{i, j}^{[l + 1]}} \pdv{z_{i, j}^{[l + 1]}}{w_{j, k}^{[l + 1]}} = \sum_i \pdv{\J}{z_{i, j}^{[l + 1]}} \tilde{a}_{i, k}^{[l]}, \\
\pdv{\J}{b_j^{[l + 1]}} &= \sum_i \pdv{\J}{z_{i, j}^{[l + 1]}} \pdv{z_{i, j}^{[l + 1]}}{b_j^{[l + 1]}} = \sum_i \pdv{\J}{z_{i, j}^{[l + 1]}}.
\end{align}
$$

$$\tilde{a}_{i, k}^{[l]}$$ affects $$z_{i, j}^{[l + 1]}$$, $$j = 1, \dots, n^{[l + 1]}$$, since

$$
\begin{align}
z_{i, 1}^{[l + 1]} &= \sum_{k = 1}^{n^{[l]}} w_{1, k}^{[l + 1]} \tilde{a}_{i, k}^{[l]} + b_1^{[l + 1]}, \\
&\vdotswithin{=} \notag \\
z_{i, j}^{[l + 1]} &= \sum_{k = 1}^{n^{[l]}} w_{j, k}^{[l + 1]} \tilde{a}_{i, k}^{[l]} + b_j^{[l + 1]}, \\
&\vdotswithin{=} \notag \\
z_{i, n^{[l + 1]}}^{[l + 1]} &= \sum_{k = 1}^{n^{[l]}} w_{n^{[l + 1]}, k}^{[l + 1]} \tilde{a}_{i, k}^{[l]} + b_{n^{[l + 1]}}^{[l + 1]};
\end{align}
$$

hence,

$$
\begin{equation}
\pdv{\J}{\tilde{a}_{i, k}^{[l]}} = \sum_j \pdv{\J}{z_{i, j}^{[l + 1]}} \pdv{z_{i, j}^{[l + 1]}}{\tilde{a}_{i, k}^{[l]}} = \sum_j \pdv{\J}{z_{i, j}^{[l + 1]}} w_{j, k}^{[l + 1]}.
\end{equation}
$$

$$a_{i, k}^{[l]}$$ only affects $$\tilde{a}_{i, k}^{[l]}$$, since

$$
\begin{equation}
\tilde{a}_{i, k}^{[l]} = \frac{r_{i, k}^{[l]}}{p_k^{[l]}} a_{i, k}^{[l]};
\end{equation}
$$

hence,

$$
\begin{equation}
\pdv{\J}{a_{i, k}^{[l]}} = \pdv{\J}{\tilde{a}_{i, k}^{[l]}} \pdv{\tilde{a}_{i, k}^{[l]}}{a_{i, k}^{[l]}} = \pdv{\J}{\tilde{a}_{i, k}^{[l]}} \frac{r_{i, k}^{[l]}}{p_k^{[l]}}.
\end{equation}
$$

In summary:

$$
\begin{align}
\pdv{\J}{a_{i, j}^{[l + 1]}} &= \text{?}, \\
\pdv{\J}{z_{i, j}^{[l + 1]}} &= \sum_{\jj = 1}^{n^{[l + 1]}} \pdv{\J}{a_{i, \jj}^{[l + 1]}} \pdv{a_{i, \jj}^{[l + 1]}}{z_{i, j}^{[l + 1]}}, \\
\pdv{\J}{w_{j, k}^{[l + 1]}} &= \sum_i \pdv{\J}{z_{i, j}^{[l + 1]}} \tilde{a}_{i, k}^{[l]}, \\
\pdv{\J}{b_j^{[l + 1]}} &= \sum_i \pdv{\J}{z_{i, j}^{[l + 1]}}, \\
\pdv{\J}{\tilde{a}_{i, k}^{[l]}} &= \sum_j \pdv{\J}{z_{i, j}^{[l + 1]}} w_{j, k}^{[l + 1]}, \\
\pdv{\J}{a_{i, k}^{[l]}} &= \pdv{\J}{\tilde{a}_{i, k}^{[l]}} \frac{r_{i, k}^{[l]}}{p_k^{[l]}}.
\end{align}
$$
