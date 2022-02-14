---
title: "Feedforward Neural Networks in Depth, Part 3: Cost Functions"
---

This post is the last of a three-part series in which we set out to derive the mathematics behind feedforward neural networks. In short, we covered forward and backward propagations in [the first post]({% post_url 2021-12-10-feedforward-neural-networks-part-1 %}){: target="_blank" }, and we worked on activation functions in [the second post]({% post_url 2021-12-21-feedforward-neural-networks-part-2 %}){: target="_blank" }. Moreover, we have not yet addressed cost functions and the backpropagation seed $$\pdv{J}{\vec{A}^{[L]}} = \pdv{J}{\vec{\hat{Y}}}$$. It is time we do that.

## Binary Classification

In binary classification, the cost function is given by

$$
\begin{equation*}
\begin{split}
J &= f(\vec{\hat{Y}}, \vec{Y}) = f(\vec{A}^{[L]}, \vec{Y}) \\
&= -\frac{1}{m} \sum_i (y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)) \\
&= -\frac{1}{m} \sum_i (y_i \log(a_i^{[L]}) + (1 - y_i) \log(1 - a_i^{[L]})),
\end{split}
\end{equation*}
$$

which we can write as

$$
\begin{equation}
J = -\frac{1}{m} \underbrace{\sum_{\text{axis} = 1} (\vec{Y} \odot \log(\vec{A}^{[L]}) + (1 - \vec{Y}) \odot \log(1 - \vec{A}^{[L]}))}_\text{scalar}.
\end{equation}
$$

Next, we construct a computation graph:

$$
\begin{align*}
u_{0, i} &= a_i^{[L]}, \\
u_{1, i} &= 1 - u_{0, i}, \\
u_{2, i} &= \log(u_{0, i}), \\
u_{3, i} &= \log(u_{1, i}), \\
u_{4, i} &= y_i u_{2, i} + (1 - y_i) u_{3, i}, \\
u_5 &= -\frac{1}{m} \sum_i u_{4, i} = J.
\end{align*}
$$

Derivative computations are now as simple as they get:

$$
\begin{align*}
\pdv{J}{u_5} &= 1, \\
\pdv{J}{u_{4, i}} &= \pdv{J}{u_5} \pdv{u_5}{u_{4, i}} = -\frac{1}{m}, \\
\pdv{J}{u_{3, i}} &= \pdv{J}{u_{4, i}} \pdv{u_{4, i}}{u_{3, i}} = -\frac{1}{m} (1 - y_i), \\
\pdv{J}{u_{2, i}} &= \pdv{J}{u_{4, i}} \pdv{u_{4, i}}{u_{2, i}} = -\frac{1}{m} y_i, \\
\pdv{J}{u_{1, i}} &= \pdv{J}{u_{3, i}} \pdv{u_{3, i}}{u_{1, i}} = -\frac{1}{m} (1 - y_i) \frac{1}{u_{1, i}} = -\frac{1}{m} \frac{1 - y_i}{1 - a_i^{[L]}}, \\
\pdv{J}{u_{0, i}} &= \pdv{J}{u_{1, i}} \pdv{u_{1, i}}{u_{0, i}} + \pdv{J}{u_{2, i}} \pdv{u_{2, i}}{u_{0, i}} \\
&= \frac{1}{m} (1 - y_i) \frac{1}{u_{1, i}} - \frac{1}{m} y_i \frac{1}{u_{0, i}} \notag \\
&= \frac{1}{m} \Bigl(\frac{1 - y_i}{1 - a_i^{[L]}} - \frac{y_i}{a_i^{[L]}}\Bigr). \notag
\end{align*}
$$

Thus,

$$
\begin{equation*}
\pdv{J}{a_i^{[L]}} = \frac{1}{m} \Bigl(\frac{1 - y_i}{1 - a_i^{[L]}} - \frac{y_i}{a_i^{[L]}}\Bigr),
\end{equation*}
$$

which implies that

$$
\begin{equation}
\pdv{J}{\vec{A}^{[L]}} = \frac{1}{m} \Bigl(\frac{1}{1 - \vec{A}^{[L]}} \odot (1 - \vec{Y}) - \frac{1}{\vec{A}^{[L]}} \odot \vec{Y}\Bigr).
\end{equation}
$$

In addition, since the sigmoid activation function is used in the output layer, we get

$$
\begin{equation*}
\begin{split}
\pdv{J}{z_i^{[L]}} &= \pdv{J}{a_i^{[L]}} a_i^{[L]} (1 - a_i^{[L]}) \\
&= \frac{1}{m} \Bigl(\frac{1 - y_i}{1 - a_i^{[L]}} - \frac{y_i}{a_i^{[L]}}\Bigr) a_i^{[L]} (1 - a_i^{[L]}) \\
&= \frac{1}{m} ((1 - y_i) a_i^{[L]} - y_i (1 - a_i^{[L]})) \\
&= \frac{1}{m} (a_i^{[L]} - y_i).
\end{split}
\end{equation*}
$$

In other words,

$$
\begin{equation}
\pdv{J}{\vec{Z}^{[L]}} = \frac{1}{m} (\vec{A}^{[L]} - \vec{Y}).
\end{equation}
$$

Note that both $$\pdv{J}{\vec{Z}^{[L]}} \in \R^{1 \times m}$$ and $$\pdv{J}{\vec{A}^{[L]}} \in \R^{1 \times m}$$, because $$n^{[L]} = 1$$ in this case.

## Multiclass Classification

In multiclass classification, the cost function is instead given by

$$
\begin{equation*}
\begin{split}
J &= f(\vec{\hat{Y}}, \vec{Y}) = f(\vec{A}^{[L]}, \vec{Y}) \\
&= -\frac{1}{m} \sum_i \sum_j y_{j, i} \log(\hat{y}_{j, i}) \\
&= -\frac{1}{m} \sum_i \sum_j y_{j, i} \log(a_{j, i}^{[L]}),
\end{split}
\end{equation*}
$$

where $$j = 1, \dots, n^{[L]}$$.

We can vectorize the cost expression:

$$
\begin{equation}
J = -\frac{1}{m} \underbrace{\sum_{\substack{\text{axis} = 0 \\ \text{axis} = 1}} \vec{Y} \odot \log(\vec{A}^{[L]})}_\text{scalar}.
\end{equation}
$$

Next, let us introduce intermediate variables:

$$
\begin{align*}
u_{0, j, i} &= a_{j, i}^{[L]}, \\
u_{1, j, i} &= \log(u_{0, j, i}), \\
u_{2, j, i} &= y_{j, i} u_{1, j, i}, \\
u_{3, i} &= \sum_j u_{2, j, i}, \\
u_4 &= -\frac{1}{m} \sum_i u_{3, i} = J.
\end{align*}
$$

With the computation graph in place, we can perform backward propagation:

$$
\begin{align*}
\pdv{J}{u_4} &= 1, \\
\pdv{J}{u_{3, i}} &= \pdv{J}{u_4} \pdv{u_4}{u_{3, i}} = -\frac{1}{m}, \\
\pdv{J}{u_{2, j, i}} &= \pdv{J}{u_{3, i}} \pdv{u_{3, i}}{u_{2, j, i}} = -\frac{1}{m}, \\
\pdv{J}{u_{1, j, i}} &= \pdv{J}{u_{2, j, i}} \pdv{u_{2, j, i}}{u_{1, j, i}} = -\frac{1}{m} y_{j, i}, \\
\pdv{J}{u_{0, j, i}} &= \pdv{J}{u_{1, j, i}} \pdv{u_{1, j, i}}{u_{0, j, i}} = -\frac{1}{m} y_{j, i} \frac{1}{u_{0, j, i}} = -\frac{1}{m} \frac{y_{j, i}}{a_{j, i}^{[L]}}.
\end{align*}
$$

Hence,

$$
\begin{equation*}
\pdv{J}{a_{j, i}^{[L]}} = -\frac{1}{m} \frac{y_{j, i}}{a_{j, i}^{[L]}}.
\end{equation*}
$$

Vectorization is trivial:

$$
\begin{equation}
\pdv{J}{\vec{A}^{[L]}} = -\frac{1}{m} \frac{1}{\vec{A}^{[L]}} \odot \vec{Y}.
\end{equation}
$$

Furthermore, since the output layer uses the softmax activation function, we get

$$
\begin{equation*}
\begin{split}
\pdv{J}{z_{j, i}^{[L]}} &= a_{j, i}^{[L]} \Bigl(\pdv{J}{a_{j, i}^{[L]}} - \sum_p \pdv{J}{a_{p, i}^{[L]}} a_{p, i}^{[L]}\Bigr) \\
&= a_{j, i}^{[L]} \Bigl(-\frac{1}{m} \frac{y_{j, i}}{a_{j, i}^{[L]}} + \sum_p \frac{1}{m} \frac{y_{p, i}}{a_{p, i}^{[L]}} a_{p, i}^{[L]}\Bigr) \\
&= \frac{1}{m} \Bigl(-y_{j, i} + a_{j, i}^{[L]} \underbrace{\sum_p y_{p, i}}_{\mathclap{\sum \text{probabilities} = 1}}\Bigr) \\
&= \frac{1}{m} (a_{j, i}^{[L]} - y_{j, i}).
\end{split}
\end{equation*}
$$

Note that $$p = 1, \dots, n^{[L]}$$.

To conclude,

$$
\begin{equation}
\pdv{J}{\vec{Z}^{[L]}} = \frac{1}{m} (\vec{A}^{[L]} - \vec{Y}).
\end{equation}
$$

## Multi-Label Classification

We can view multi-label classification as $$j$$ binary classification problems:

$$
\begin{equation*}
\begin{split}
J &= f(\vec{\hat{Y}}, \vec{Y}) = f(\vec{A}^{[L]}, \vec{Y}) \\
&= \sum_j \Bigl(-\frac{1}{m} \sum_i (y_{j, i} \log(\hat{y}_{j, i}) + (1 - y_{j, i}) \log(1 - \hat{y}_{j, i}))\Bigr) \\
&= \sum_j \Bigl(-\frac{1}{m} \sum_i (y_{j, i} \log(a_{j, i}^{[L]}) + (1 - y_{j, i}) \log(1 - a_{j, i}^{[L]}))\Bigr),
\end{split}
\end{equation*}
$$

where once again $$j = 1, \dots, n^{[L]}$$.

Vectorization gives

$$
\begin{equation}
J = -\frac{1}{m} \underbrace{\sum_{\substack{\text{axis} = 1 \\ \text{axis} = 0}} (\vec{Y} \odot \log(\vec{A}^{[L]}) + (1 - \vec{Y}) \odot \log(1 - \vec{A}^{[L]}))}_\text{scalar}.
\end{equation}
$$

It is no coincidence that the following computation graph resembles the one we constructed for binary classification:

$$
\begin{align*}
u_{0, j, i} &= a_{j, i}^{[L]}, \\
u_{1, j, i} &= 1 - u_{0, j, i}, \\
u_{2, j, i} &= \log(u_{0, j, i}), \\
u_{3, j, i} &= \log(u_{1, j, i}), \\
u_{4, j, i} &= y_{j, i} u_{2, j, i} + (1 - y_{j, i}) u_{3, j, i}, \\
u_{5, j} &= -\frac{1}{m} \sum_i u_{4, j, i}, \\
u_6 &= \sum_j u_{5, j} = J.
\end{align*}
$$

Next, we compute the partial derivatives:

$$
\begin{align*}
\pdv{J}{u_6} &= 1, \\
\pdv{J}{u_{5, j}} &= \pdv{J}{u_6} \pdv{u_6}{u_{5, j}} = 1, \\
\pdv{J}{u_{4, j, i}} &= \pdv{J}{u_{5, j}} \pdv{u_{5, j}}{u_{4, j, i}} = -\frac{1}{m}, \\
\pdv{J}{u_{3, j, i}} &= \pdv{J}{u_{4, j, i}} \pdv{u_{4, j, i}}{u_{3, j, i}} = -\frac{1}{m} (1 - y_{j, i}), \\
\pdv{J}{u_{2, j, i}} &= \pdv{J}{u_{4, j, i}} \pdv{u_{4, j, i}}{u_{2, j, i}} = -\frac{1}{m} y_{j, i}, \\
\pdv{J}{u_{1, j, i}} &= \pdv{J}{u_{3, j, i}} \pdv{u_{3, j, i}}{u_{1, j, i}} = -\frac{1}{m} (1 - y_{j, i}) \frac{1}{u_{1, j, i}} = -\frac{1}{m} \frac{1 - y_{j, i}}{1 - a_{j, i}^{[L]}}, \\
\pdv{J}{u_{0, j, i}} &= \pdv{J}{u_{1, j, i}} \pdv{u_{1, j, i}}{u_{0, j, i}} + \pdv{J}{u_{2, j, i}} \pdv{u_{2, j, i}}{u_{0, j, i}} \\
&= \frac{1}{m} (1 - y_{j, i}) \frac{1}{u_{1, j, i}} - \frac{1}{m} y_{j, i} \frac{1}{u_{0, j, i}} \notag \\
&= \frac{1}{m} \Bigl(\frac{1 - y_{j, i}}{1 - a_{j, i}^{[L]}} - \frac{y_{j, i}}{a_{j, i}^{[L]}}\Bigr). \notag
\end{align*}
$$

Simply put, we have

$$
\begin{equation*}
\pdv{J}{a_{j, i}^{[L]}} = \frac{1}{m} \Bigl(\frac{1 - y_{j, i}}{1 - a_{j, i}^{[L]}} - \frac{y_{j, i}}{a_{j, i}^{[L]}}\Bigr),
\end{equation*}
$$

and

$$
\begin{equation}
\pdv{J}{\vec{A}^{[L]}} = \frac{1}{m} \Bigl(\frac{1}{1 - \vec{A}^{[L]}} \odot (1 - \vec{Y}) - \frac{1}{\vec{A}^{[L]}} \odot \vec{Y}\Bigr).
\end{equation}
$$

Bearing in mind that we view multi-label classification as $$j$$ binary classification problems, we also know that the output layer uses the sigmoid activation function. As a result,

$$
\begin{equation*}
\begin{split}
\pdv{J}{z_{j, i}^{[L]}} &= \pdv{J}{a_{j, i}^{[L]}} a_{j, i}^{[L]} (1 - a_{j, i}^{[L]}) \\
&= \frac{1}{m} \Bigl(\frac{1 - y_{j, i}}{1 - a_{j, i}^{[L]}} - \frac{y_{j, i}}{a_{j, i}^{[L]}}\Bigr) a_{j, i}^{[L]} (1 - a_{j, i}^{[L]}) \\
&= \frac{1}{m} ((1 - y_{j, i}) a_{j, i}^{[L]} - y_{j, i} (1 - a_{j, i}^{[L]})) \\
&= \frac{1}{m} (a_{j, i}^{[L]} - y_{j, i}),
\end{split}
\end{equation*}
$$

which we can vectorize as

$$
\begin{equation}
\pdv{J}{\vec{Z}^{[L]}} = \frac{1}{m} (\vec{A}^{[L]} - \vec{Y}).
\end{equation}
$$
