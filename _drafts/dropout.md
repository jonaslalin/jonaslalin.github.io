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
\tilde{a}_{i, k}^{[l]} &= r_{i, k}^{[l]} \frac{1}{p_k^{[l]}} a_{i, k}^{[l]}, \\
z_{i, j}^{[l + 1]} &= \sum_{k = 1}^{n^{[l]}} w_{k, j}^{[l + 1]} \tilde{a}_{i, k}^{[l]} + b_j^{[l + 1]}, \\
a_{i, j}^{[l + 1]} &= g_j^{[l + 1]}(z_{i, 1}^{[l + 1]}, \dots, z_{i, j}^{[l + 1]}, \dots, z_{i, n^{[l + 1]}}^{[l + 1]}).
\end{align}
$$

$$
\begin{align}
\vec{A}^{[l]} &=
\begin{bmatrix}
a_{1, 1}^{[l]} &\dots &a_{1, k}^{[l]} &\dots &a_{1, n^{[l]}}^{[l]} \\
\vdots &\ddots &\vdots &\ddots &\vdots \\
a_{i, 1}^{[l]} &\dots &a_{i, k}^{[l]} &\dots &a_{i, n^{[l]}}^{[l]} \\
\vdots &\ddots &\vdots &\ddots &\vdots \\
a_{m, 1}^{[l]} &\dots &a_{m, k}^{[l]} &\dots &a_{m, n^{[l]}}^{[l]}
\end{bmatrix}, \\
\vec{R}^{[l]} &=
\begin{bmatrix}
r_{1, 1}^{[l]} &\dots &r_{1, k}^{[l]} &\dots &r_{1, n^{[l]}}^{[l]} \\
\vdots &\ddots &\vdots &\ddots &\vdots \\
r_{i, 1}^{[l]} &\dots &r_{i, k}^{[l]} &\dots &r_{i, n^{[l]}}^{[l]} \\
\vdots &\ddots &\vdots &\ddots &\vdots \\
r_{m, 1}^{[l]} &\dots &r_{m, k}^{[l]} &\dots &r_{m, n^{[l]}}^{[l]}
\end{bmatrix}, \\
\vec{p}^{[l]} &=
\begin{bmatrix}
p_1^{[l]} &\dots &p_k^{[l]} &\dots &p_{n^{[l]}}^{[l]}
\end{bmatrix}, \\
\vec{\tilde{A}}^{[l]} &=
\begin{bmatrix}
\tilde{a}_{1, 1}^{[l]} &\dots &\tilde{a}_{1, k}^{[l]} &\dots &\tilde{a}_{1, n^{[l]}}^{[l]} \\
\vdots &\ddots &\vdots &\ddots &\vdots \\
\tilde{a}_{i, 1}^{[l]} &\dots &\tilde{a}_{i, k}^{[l]} &\dots &\tilde{a}_{i, n^{[l]}}^{[l]} \\
\vdots &\ddots &\vdots &\ddots &\vdots \\
\tilde{a}_{m, 1}^{[l]} &\dots &\tilde{a}_{m, k}^{[l]} &\dots &\tilde{a}_{m, n^{[l]}}^{[l]}
\end{bmatrix}, \\
\vec{z}_i^{[l + 1]} &=
\begin{bmatrix}
z_{i, 1}^{[l + 1]} &\dots &z_{i, j}^{[l + 1]} &\dots &z_{i, n^{[l + 1]}}^{[l + 1]}
\end{bmatrix}, \\
\vec{Z}^{[l + 1]} &=
\begin{bmatrix}
\vec{z}_1^{[l + 1]} \\
\vdots \\
\vec{z}_i^{[l + 1]} \\
\vdots \\
\vec{z}_m^{[l + 1]}
\end{bmatrix} \\
&=
\begin{bmatrix}
z_{1, 1}^{[l + 1]} &\dots &z_{1, j}^{[l + 1]} &\dots &z_{1, n^{[l + 1]}}^{[l + 1]} \\
\vdots &\ddots &\vdots &\ddots &\vdots \\
z_{i, 1}^{[l + 1]} &\dots &z_{i, j}^{[l + 1]} &\dots &z_{i, n^{[l + 1]}}^{[l + 1]} \\
\vdots &\ddots &\vdots &\ddots &\vdots \\
z_{m, 1}^{[l + 1]} &\dots &z_{m, j}^{[l + 1]} &\dots &z_{m, n^{[l + 1]}}^{[l + 1]}
\end{bmatrix}, \notag \\
\vec{W}^{[l + 1]} &=
\begin{bmatrix}
w_{1, 1}^{[l + 1]} &\dots &w_{1, j}^{[l + 1]} &\dots &w_{1, n^{[l + 1]}}^{[l + 1]} \\
\vdots &\ddots &\vdots &\ddots &\vdots \\
w_{k, 1}^{[l + 1]} &\dots &w_{k, j}^{[l + 1]} &\dots &w_{k, n^{[l + 1]}}^{[l + 1]} \\
\vdots &\ddots &\vdots &\ddots &\vdots \\
w_{n^{[l]}, 1}^{[l + 1]} &\dots &w_{n^{[l]}, j}^{[l + 1]} &\dots &w_{n^{[l]}, n^{[l + 1]}}^{[l + 1]}
\end{bmatrix}, \\
\vec{b}^{[l + 1]} &=
\begin{bmatrix}
b_1^{[l + 1]} &\dots &b_j^{[l + 1]} &\dots &b_{n^{[l + 1]}}^{[l + 1]}
\end{bmatrix}, \\
\vec{a}_i^{[l + 1]} &=
\begin{bmatrix}
a_{i, 1}^{[l + 1]} &\dots &a_{i, j}^{[l + 1]} &\dots &a_{i, n^{[l + 1]}}^{[l + 1]}
\end{bmatrix}, \\
\vec{A}^{[l + 1]} &=
\begin{bmatrix}
\vec{a}_1^{[l + 1]} \\
\vdots \\
\vec{a}_i^{[l + 1]} \\
\vdots \\
\vec{a}_m^{[l + 1]}
\end{bmatrix} \\
&=
\begin{bmatrix}
a_{1, 1}^{[l + 1]} &\dots &a_{1, j}^{[l + 1]} &\dots &a_{1, n^{[l + 1]}}^{[l + 1]} \\
\vdots &\ddots &\vdots &\ddots &\vdots \\
a_{i, 1}^{[l + 1]} &\dots &a_{i, j}^{[l + 1]} &\dots &a_{i, n^{[l + 1]}}^{[l + 1]} \\
\vdots &\ddots &\vdots &\ddots &\vdots \\
a_{m, 1}^{[l + 1]} &\dots &a_{m, j}^{[l + 1]} &\dots &a_{m, n^{[l + 1]}}^{[l + 1]}
\end{bmatrix}, \notag \\
\vec{g}^{[l + 1]} &\colon \R^{n^{[l + 1]}} \to \R^{n^{[l + 1]}}.
\end{align}
$$

$$
\begin{align}
\vec{A}^{[l]} &= \text{?}, \\
\vec{\tilde{A}}^{[l]} &= \vec{R}^{[l]} \odot \frac{1}{\broadcast(\vec{p}^{[l]})} \odot \vec{A}^{[l]}, \\
\vec{Z}^{[l + 1]} &= \vec{\tilde{A}}^{[l]} \vec{W}^{[l + 1]} + \broadcast(\vec{b}^{[l + 1]}), \\
\vec{A}^{[l + 1]} &= \vec{g}^{[l + 1]}(\vec{Z}^{[l + 1]}).
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

Both $$w_{k, j}^{[l + 1]}$$ and $$b_j^{[l + 1]}$$ affect $$z_{i, j}^{[l + 1]}$$, $$i = 1, \dots, m$$, since

$$
\begin{align}
z_{1, j}^{[l + 1]} &= \sum_{k = 1}^{n^{[l]}} w_{k, j}^{[l + 1]} \tilde{a}_{1, k}^{[l]} + b_j^{[l + 1]}, \\
&\vdotswithin{=} \notag \\
z_{i, j}^{[l + 1]} &= \sum_{k = 1}^{n^{[l]}} w_{k, j}^{[l + 1]} \tilde{a}_{i, k}^{[l]} + b_j^{[l + 1]}, \\
&\vdotswithin{=} \notag \\
z_{m, j}^{[l + 1]} &= \sum_{k = 1}^{n^{[l]}} w_{k, j}^{[l + 1]} \tilde{a}_{m, k}^{[l]} + b_j^{[l + 1]};
\end{align}
$$

hence,

$$
\begin{align}
\pdv{\J}{w_{k, j}^{[l + 1]}} &= \sum_{i = 1}^m \pdv{\J}{z_{i, j}^{[l + 1]}} \pdv{z_{i, j}^{[l + 1]}}{w_{k, j}^{[l + 1]}} = \sum_{i = 1}^m \pdv{\J}{z_{i, j}^{[l + 1]}} \tilde{a}_{i, k}^{[l]}, \\
\pdv{\J}{b_j^{[l + 1]}} &= \sum_{i = 1}^m \pdv{\J}{z_{i, j}^{[l + 1]}} \pdv{z_{i, j}^{[l + 1]}}{b_j^{[l + 1]}} = \sum_{i = 1}^m \pdv{\J}{z_{i, j}^{[l + 1]}}.
\end{align}
$$

$$\tilde{a}_{i, k}^{[l]}$$ affects $$z_{i, j}^{[l + 1]}$$, $$j = 1, \dots, n^{[l + 1]}$$, since

$$
\begin{align}
z_{i, 1}^{[l + 1]} &= \sum_{k = 1}^{n^{[l]}} w_{k, 1}^{[l + 1]} \tilde{a}_{i, k}^{[l]} + b_1^{[l + 1]}, \\
&\vdotswithin{=} \notag \\
z_{i, j}^{[l + 1]} &= \sum_{k = 1}^{n^{[l]}} w_{k, j}^{[l + 1]} \tilde{a}_{i, k}^{[l]} + b_j^{[l + 1]}, \\
&\vdotswithin{=} \notag \\
z_{i, n^{[l + 1]}}^{[l + 1]} &= \sum_{k = 1}^{n^{[l]}} w_{k, n^{[l + 1]}}^{[l + 1]} \tilde{a}_{i, k}^{[l]} + b_{n^{[l + 1]}}^{[l + 1]};
\end{align}
$$

hence,

$$
\begin{equation}
\pdv{\J}{\tilde{a}_{i, k}^{[l]}} = \sum_{j = 1}^{n^{[l + 1]}} \pdv{\J}{z_{i, j}^{[l + 1]}} \pdv{z_{i, j}^{[l + 1]}}{\tilde{a}_{i, k}^{[l]}} = \sum_{j = 1}^{n^{[l + 1]}} \pdv{\J}{z_{i, j}^{[l + 1]}} w_{k, j}^{[l + 1]}.
\end{equation}
$$

$$a_{i, k}^{[l]}$$ only affects $$\tilde{a}_{i, k}^{[l]}$$, since

$$
\begin{equation}
\tilde{a}_{i, k}^{[l]} = r_{i, k}^{[l]} \frac{1}{p_k^{[l]}} a_{i, k}^{[l]};
\end{equation}
$$

hence,

$$
\begin{equation}
\pdv{\J}{a_{i, k}^{[l]}} = \pdv{\J}{\tilde{a}_{i, k}^{[l]}} \pdv{\tilde{a}_{i, k}^{[l]}}{a_{i, k}^{[l]}} = \pdv{\J}{\tilde{a}_{i, k}^{[l]}} r_{i, k}^{[l]} \frac{1}{p_k^{[l]}}.
\end{equation}
$$

In summary:

$$
\begin{align}
\pdv{\J}{a_{i, j}^{[l + 1]}} &= \text{?}, \\
\pdv{\J}{z_{i, j}^{[l + 1]}} &= \sum_{\jj = 1}^{n^{[l + 1]}} \pdv{\J}{a_{i, \jj}^{[l + 1]}} \pdv{a_{i, \jj}^{[l + 1]}}{z_{i, j}^{[l + 1]}}, \\
\pdv{\J}{w_{k, j}^{[l + 1]}} &= \sum_{i = 1}^m \pdv{\J}{z_{i, j}^{[l + 1]}} \tilde{a}_{i, k}^{[l]}, \\
\pdv{\J}{b_j^{[l + 1]}} &= \sum_{i = 1}^m \pdv{\J}{z_{i, j}^{[l + 1]}}, \\
\pdv{\J}{\tilde{a}_{i, k}^{[l]}} &= \sum_{j = 1}^{n^{[l + 1]}} \pdv{\J}{z_{i, j}^{[l + 1]}} w_{k, j}^{[l + 1]}, \\
\pdv{\J}{a_{i, k}^{[l]}} &= \pdv{\J}{\tilde{a}_{i, k}^{[l]}} r_{i, k}^{[l]} \frac{1}{p_k^{[l]}}.
\end{align}
$$

$$
\begin{align}
\pdv{\J}{\vec{a}_i^{[l + 1]}} &=
\begin{bmatrix}
\dpdv{\J}{a_{i, 1}^{[l + 1]}} &\dots &\dpdv{\J}{a_{i, j}^{[l + 1]}} &\dots &\dpdv{\J}{a_{i, n^{[l + 1]}}^{[l + 1]}}
\end{bmatrix}, \\
\pdv{\J}{\vec{A}^{[l + 1]}} &=
\begin{bmatrix}
\dpdv{\J}{\vec{a}_1^{[l + 1]}} \\
\vdots \\
\dpdv{\J}{\vec{a}_i^{[l + 1]}} \\
\vdots \\
\dpdv{\J}{\vec{a}_m^{[l + 1]}}
\end{bmatrix} \\
&=
\begin{bmatrix}
\dpdv{\J}{a_{1, 1}^{[l + 1]}} &\dots &\dpdv{\J}{a_{1, j}^{[l + 1]}} &\dots &\dpdv{\J}{a_{1, n^{[l + 1]}}^{[l + 1]}} \\
\vdots &\ddots &\vdots &\ddots &\vdots \\
\dpdv{\J}{a_{i, 1}^{[l + 1]}} &\dots &\dpdv{\J}{a_{i, j}^{[l + 1]}} &\dots &\dpdv{\J}{a_{i, n^{[l + 1]}}^{[l + 1]}} \\
\vdots &\ddots &\vdots &\ddots &\vdots \\
\dpdv{\J}{a_{m, 1}^{[l + 1]}} &\dots &\dpdv{\J}{a_{m, j}^{[l + 1]}} &\dots &\dpdv{\J}{a_{m, n^{[l + 1]}}^{[l + 1]}}
\end{bmatrix}, \notag \\
\pdv{\J}{\vec{z}_i^{[l + 1]}} &=
\begin{bmatrix}
\dpdv{\J}{z_{i, 1}^{[l + 1]}} &\dots &\dpdv{\J}{z_{i, j}^{[l + 1]}} &\dots &\dpdv{\J}{z_{i, n^{[l + 1]}}^{[l + 1]}}
\end{bmatrix}, \\
\pdv{\J}{\vec{Z}^{[l + 1]}} &=
\begin{bmatrix}
\dpdv{\J}{\vec{z}_1^{[l + 1]}} \\
\vdots \\
\dpdv{\J}{\vec{z}_i^{[l + 1]}} \\
\vdots \\
\dpdv{\J}{\vec{z}_m^{[l + 1]}}
\end{bmatrix} \\
&=
\begin{bmatrix}
\dpdv{\J}{z_{1, 1}^{[l + 1]}} &\dots &\dpdv{\J}{z_{1, j}^{[l + 1]}} &\dots &\dpdv{\J}{z_{1, n^{[l + 1]}}^{[l + 1]}} \\
\vdots &\ddots &\vdots &\ddots &\vdots \\
\dpdv{\J}{z_{i, 1}^{[l + 1]}} &\dots &\dpdv{\J}{z_{i, j}^{[l + 1]}} &\dots &\dpdv{\J}{z_{i, n^{[l + 1]}}^{[l + 1]}} \\
\vdots &\ddots &\vdots &\ddots &\vdots \\
\dpdv{\J}{z_{m, 1}^{[l + 1]}} &\dots &\dpdv{\J}{z_{m, j}^{[l + 1]}} &\dots &\dpdv{\J}{z_{m, n^{[l + 1]}}^{[l + 1]}}
\end{bmatrix}, \notag \\
\pdv{\J}{\vec{W}^{[l + 1]}} &=
\begin{bmatrix}
\dpdv{\J}{w_{1, 1}^{[l + 1]}} &\dots &\dpdv{\J}{w_{1, j}^{[l + 1]}} &\dots &\dpdv{\J}{w_{1, n^{[l + 1]}}^{[l + 1]}} \\
\vdots &\ddots &\vdots &\ddots &\vdots \\
\dpdv{\J}{w_{k, 1}^{[l + 1]}} &\dots &\dpdv{\J}{w_{k, j}^{[l + 1]}} &\dots &\dpdv{\J}{w_{k, n^{[l + 1]}}^{[l + 1]}} \\
\vdots &\ddots &\vdots &\ddots &\vdots \\
\dpdv{\J}{w_{n^{[l]}, 1}^{[l + 1]}} &\dots &\dpdv{\J}{w_{n^{[l]}, j}^{[l + 1]}} &\dots &\dpdv{\J}{w_{n^{[l]}, n^{[l + 1]}}^{[l + 1]}}
\end{bmatrix}, \\
\pdv{\J}{\vec{b}^{[l + 1]}} &=
\begin{bmatrix}
\dpdv{\J}{b_1^{[l + 1]}} &\dots &\dpdv{\J}{b_j^{[l + 1]}} &\dots &\dpdv{\J}{b_{n^{[l + 1]}}^{[l + 1]}}
\end{bmatrix}, \\
\pdv{\J}{\vec{\tilde{A}}^{[l]}} &=
\begin{bmatrix}
\dpdv{\J}{\tilde{a}_{1, 1}^{[l]}} &\dots &\dpdv{\J}{\tilde{a}_{1, k}^{[l]}} &\dots &\dpdv{\J}{\tilde{a}_{1, n^{[l]}}^{[l]}} \\
\vdots &\ddots &\vdots &\ddots &\vdots \\
\dpdv{\J}{\tilde{a}_{i, 1}^{[l]}} &\dots &\dpdv{\J}{\tilde{a}_{i, k}^{[l]}} &\dots &\dpdv{\J}{\tilde{a}_{i, n^{[l]}}^{[l]}} \\
\vdots &\ddots &\vdots &\ddots &\vdots \\
\dpdv{\J}{\tilde{a}_{m, 1}^{[l]}} &\dots &\dpdv{\J}{\tilde{a}_{m, k}^{[l]}} &\dots &\dpdv{\J}{\tilde{a}_{m, n^{[l]}}^{[l]}}
\end{bmatrix}, \\
\pdv{\J}{\vec{A}^{[l]}} &=
\begin{bmatrix}
\dpdv{\J}{a_{1, 1}^{[l]}} &\dots &\dpdv{\J}{a_{1, k}^{[l]}} &\dots &\dpdv{\J}{a_{1, n^{[l]}}^{[l]}} \\
\vdots &\ddots &\vdots &\ddots &\vdots \\
\dpdv{\J}{a_{i, 1}^{[l]}} &\dots &\dpdv{\J}{a_{i, k}^{[l]}} &\dots &\dpdv{\J}{a_{i, n^{[l]}}^{[l]}} \\
\vdots &\ddots &\vdots &\ddots &\vdots \\
\dpdv{\J}{a_{m, 1}^{[l]}} &\dots &\dpdv{\J}{a_{m, k}^{[l]}} &\dots &\dpdv{\J}{a_{m, n^{[l]}}^{[l]}}
\end{bmatrix}.
\end{align}
$$

$$
\begin{equation}
\begin{split}
\pdv{\J}{\vec{z}_i^{[l + 1]}} &=
\begin{bmatrix}
\dpdv{\J}{z_{i, 1}^{[l + 1]}} &\dots &\dpdv{\J}{z_{i, j}^{[l + 1]}} &\dots &\dpdv{\J}{z_{i, n^{[l + 1]}}^{[l + 1]}}
\end{bmatrix} \\
&=
\begin{bmatrix}
\dpdv{\J}{a_{i, 1}^{[l + 1]}} &\dots &\dpdv{\J}{a_{i, j}^{[l + 1]}} &\dots &\dpdv{\J}{a_{i, n^{[l + 1]}}^{[l + 1]}}
\end{bmatrix} \\
&\peq {} \cdot
\begin{bmatrix}
\dpdv{a_{i, 1}^{[l + 1]}}{z_{i, 1}^{[l + 1]}} &\dots &\dpdv{a_{i, 1}^{[l + 1]}}{z_{i, j}^{[l + 1]}} &\dots &\dpdv{a_{i, 1}^{[l + 1]}}{z_{i, n^{[l + 1]}}^{[l + 1]}} \\
\vdots &\ddots &\vdots &\ddots &\vdots \\
\dpdv{a_{i, j}^{[l + 1]}}{z_{i, 1}^{[l + 1]}} &\dots &\dpdv{a_{i, j}^{[l + 1]}}{z_{i, j}^{[l + 1]}} &\dots &\dpdv{a_{i, j}^{[l + 1]}}{z_{i, n^{[l + 1]}}^{[l + 1]}} \\
\vdots &\ddots &\vdots &\ddots &\vdots \\
\dpdv{a_{i, n^{[l + 1]}}^{[l + 1]}}{z_{i, 1}^{[l + 1]}} &\dots &\dpdv{a_{i, n^{[l + 1]}}^{[l + 1]}}{z_{i, j}^{[l + 1]}} &\dots &\dpdv{a_{i, n^{[l + 1]}}^{[l + 1]}}{z_{i, n^{[l + 1]}}^{[l + 1]}}
\end{bmatrix} \\
&= \pdv{\J}{\vec{a}_i^{[l + 1]}} \pdv{\vec{a}_i^{[l + 1]}}{\vec{z}_i^{[l + 1]}}.
\end{split}
\end{equation}
$$

In summary:

$$
\begin{align}
\pdv{\J}{\vec{A}^{[l + 1]}} &= \text{?}, \\
\pdv{\J}{\vec{z}_i^{[l + 1]}} &= \pdv{\J}{\vec{a}_i^{[l + 1]}} \pdv{\vec{a}_i^{[l + 1]}}{\vec{z}_i^{[l + 1]}}, \\
\pdv{\J}{\vec{Z}^{[l + 1]}} &=
\begin{bmatrix}
\dpdv{\J}{\vec{z}_1^{[l + 1]}} \\
\vdots \\
\dpdv{\J}{\vec{z}_i^{[l + 1]}} \\
\vdots \\
\dpdv{\J}{\vec{z}_m^{[l + 1]}}
\end{bmatrix}, \\
\pdv{\J}{\vec{W}^{[l + 1]}} &= {\vec{\tilde{A}}^{[l]}}^\T \pdv{\J}{\vec{Z}^{[l + 1]}}, \\
\pdv{\J}{\vec{b}^{[l + 1]}} &= \sum_{i = 1}^m \pdv{\J}{\vec{z}_i^{[l + 1]}}, \\
\pdv{\J}{\vec{\tilde{A}}^{[l]}} &= \pdv{\J}{\vec{Z}^{[l + 1]}} {\vec{W}^{[l + 1]}}^\T, \\
\pdv{\J}{\vec{A}^{[l]}} &= \pdv{\J}{\vec{\tilde{A}}^{[l]}} \odot \vec{R}^{[l]} \odot \frac{1}{\broadcast(\vec{p}^{[l]})}.
\end{align}
$$
