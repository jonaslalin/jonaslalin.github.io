---
title: "Feedforward Neural Networks in Depth, Part 2: Activation Functions"
---

This is the second post of a three-part series in which we derive the mathematics behind feedforward neural networks. We worked our way through forward and backward propagations in [the first post]({% post_url 2021-12-10-feedforward-neural-networks-part-1 %}){: target="_blank" }, but if you remember, we only mentioned activation functions in passing. In particular, we did not derive an analytic expression for $$\pdv{a_{j, i}^{[l]}}{z_{j, i}^{[l]}}$$ or, by extension, $$\pdv{J}{z_{j, i}^{[l]}}$$. So let us pick up the derivations where we left off.

## ReLU

The rectified linear unit, or ReLU for short, is given by

$$
\begin{equation*}
\begin{split}
a_{j, i}^{[l]} &= g_j^{[l]}(z_{1, i}^{[l]}, \dots, z_{j, i}^{[l]}, \dots, z_{n^{[l]}, i}^{[l]}) \\
&= \max(0, z_{j, i}^{[l]}) \\
&=
\begin{cases}
z_{j, i}^{[l]} &\text{if } z_{j, i}^{[l]} > 0, \\
0 &\text{otherwise.}
\end{cases}
\end{split}
\end{equation*}
$$

In other words,

$$
\begin{equation}
\vec{A}^{[l]} = \max(0, \vec{Z}^{[l]}).
\end{equation}
$$

Next, we compute the partial derivatives of the activations in the current layer:

$$
\begin{align*}
\pdv{a_{j, i}^{[l]}}{z_{j, i}^{[l]}} &\coloneqq
\begin{cases}
1 &\text{if } z_{j, i}^{[l]} > 0, \\
0 &\text{otherwise,}
\end{cases} \\
&= I(z_{j, i}^{[l]} > 0), \notag \\
\pdv{a_{\jj, i}^{[l]}}{z_{j, i}^{[l]}} &= 0, \quad \forall \jj \ne j.
\end{align*}
$$

It follows that

$$
\begin{equation*}
\begin{split}
\pdv{J}{z_{j, i}^{[l]}} &= \sum_\jj \pdv{J}{a_{\jj, i}^{[l]}} \pdv{a_{\jj, i}^{[l]}}{z_{j, i}^{[l]}} \\
&= \pdv{J}{a_{j, i}^{[l]}} \pdv{a_{j, i}^{[l]}}{z_{j, i}^{[l]}} + \sum_{\jj \ne j} \pdv{J}{a_{\jj, i}^{[l]}} \pdv{a_{\jj, i}^{[l]}}{z_{j, i}^{[l]}} \\
&= \pdv{J}{a_{j, i}^{[l]}} I(z_{j, i}^{[l]} > 0),
\end{split}
\end{equation*}
$$

which we can vectorize as

$$
\begin{equation}
\pdv{J}{\vec{Z}^{[l]}} = \pdv{J}{\vec{A}^{[l]}} \odot I(\vec{Z}^{[l]} > 0),
\end{equation}
$$

where $$\odot$$ denotes element-wise multiplication.

## Sigmoid

The sigmoid activation function is given by

$$
\begin{equation*}
\begin{split}
a_{j, i}^{[l]} &= g_j^{[l]}(z_{1, i}^{[l]}, \dots, z_{j, i}^{[l]}, \dots, z_{n^{[l]}, i}^{[l]}) \\
&= \sigma(z_{j, i}^{[l]}) \\
&= \frac{1}{1 + \exp(-z_{j, i}^{[l]})}.
\end{split}
\end{equation*}
$$

Vectorization yields

$$
\begin{equation}
\vec{A}^{[l]} = \frac{1}{1 + \exp(-\vec{Z}^{[l]})}.
\end{equation}
$$

To practice backward propagation, first, we construct a computation graph:

$$
\begin{align*}
u_0 &= z_{j, i}^{[l]}, \\
u_1 &= -u_0, \\
u_2 &= \exp(u_1), \\
u_3 &= 1 + u_2, \\
u_4 &= \frac{1}{u_3} = a_{j, i}^{[l]}.
\end{align*}
$$

Then, we perform an outside first traversal of the chain rule:

$$
\begin{align*}
\pdv{a_{j, i}^{[l]}}{u_4} &= 1, \\
\pdv{a_{j, i}^{[l]}}{u_3} &= \pdv{a_{j, i}^{[l]}}{u_4} \pdv{u_4}{u_3} = -\frac{1}{u_3^2} = -\frac{1}{(1 + \exp(-z_{j, i}^{[l]}))^2}, \\
\pdv{a_{j, i}^{[l]}}{u_2} &= \pdv{a_{j, i}^{[l]}}{u_3} \pdv{u_3}{u_2} = -\frac{1}{u_3^2} = -\frac{1}{(1 + \exp(-z_{j, i}^{[l]}))^2}, \\
\pdv{a_{j, i}^{[l]}}{u_1} &= \pdv{a_{j, i}^{[l]}}{u_2} \pdv{u_2}{u_1} = -\frac{1}{u_3^2} \exp(u_1) = -\frac{\exp(-z_{j, i}^{[l]})}{(1 + \exp(-z_{j, i}^{[l]}))^2}, \\
\pdv{a_{j, i}^{[l]}}{u_0} &= \pdv{a_{j, i}^{[l]}}{u_1} \pdv{u_1}{u_0} = \frac{1}{u_3^2} \exp(u_1) = \frac{\exp(-z_{j, i}^{[l]})}{(1 + \exp(-z_{j, i}^{[l]}))^2}.
\end{align*}
$$

Let us simplify:

$$
\begin{equation*}
\begin{split}
\pdv{a_{j, i}^{[l]}}{z_{j, i}^{[l]}} &= \frac{\exp(-z_{j, i}^{[l]})}{(1 + \exp(-z_{j, i}^{[l]}))^2} \\
&= \frac{1 + \exp(-z_{j, i}^{[l]}) - 1}{(1 + \exp(-z_{j, i}^{[l]}))^2} \notag \\
&= \frac{1}{1 + \exp(-z_{j, i}^{[l]})} - \frac{1}{(1 + \exp(-z_{j, i}^{[l]}))^2} \notag \\
&= a_{j, i}^{[l]} (1 - a_{j, i}^{[l]}).
\end{split}
\end{equation*}
$$

We also note that

$$
\begin{equation*}
\pdv{a_{\jj, i}^{[l]}}{z_{j, i}^{[l]}} = 0, \quad \forall \jj \ne j.
\end{equation*}
$$

Consequently,

$$
\begin{equation*}
\begin{split}
\pdv{J}{z_{j, i}^{[l]}} &= \sum_\jj \pdv{J}{a_{\jj, i}^{[l]}} \pdv{a_{\jj, i}^{[l]}}{z_{j, i}^{[l]}} \\
&= \pdv{J}{a_{j, i}^{[l]}} \pdv{a_{j, i}^{[l]}}{z_{j, i}^{[l]}} + \sum_{\jj \ne j} \pdv{J}{a_{\jj, i}^{[l]}} \pdv{a_{\jj, i}^{[l]}}{z_{j, i}^{[l]}} \\
&= \pdv{J}{a_{j, i}^{[l]}} a_{j, i}^{[l]} (1 - a_{j, i}^{[l]}).
\end{split}
\end{equation*}
$$

Lastly, no summations mean trivial vectorization:

$$
\begin{equation}
\pdv{J}{\vec{Z}^{[l]}} = \pdv{J}{\vec{A}^{[l]}} \odot \vec{A}^{[l]} \odot (1 - \vec{A}^{[l]}).
\end{equation}
$$

## Tanh

The hyperbolic tangent function, i.e., the tanh activation function, is given by

$$
\begin{equation*}
\begin{split}
a_{j, i}^{[l]} &= g_j^{[l]}(z_{1, i}^{[l]}, \dots, z_{j, i}^{[l]}, \dots, z_{n^{[l]}, i}^{[l]}) \\
&= \tanh(z_{j, i}^{[l]}) \\
&= \frac{\exp(z_{j, i}^{[l]}) - \exp(-z_{j, i}^{[l]})}{\exp(z_{j, i}^{[l]}) + \exp(-z_{j, i}^{[l]})}.
\end{split}
\end{equation*}
$$

By utilizing element-wise multiplication, we get

$$
\begin{equation}
\vec{A}^{[l]} = \frac{1}{\exp(\vec{Z}^{[l]}) + \exp(-\vec{Z}^{[l]})} \odot (\exp(\vec{Z}^{[l]}) - \exp(-\vec{Z}^{[l]})).
\end{equation}
$$

Once again, let us introduce intermediate variables to practice backward propagation:

$$
\begin{align*}
u_0 &= z_{j, i}^{[l]}, \\
u_1 &= -u_0, \\
u_2 &= \exp(u_0), \\
u_3 &= \exp(u_1), \\
u_4 &= u_2 - u_3, \\
u_5 &= u_2 + u_3, \\
u_6 &= \frac{1}{u_5}, \\
u_7 &= u_4 u_6 = a_{j, i}^{[l]}.
\end{align*}
$$

Next, we compute the partial derivatives:

$$
\begin{align*}
\pdv{a_{j, i}^{[l]}}{u_7} &= 1, \\
\pdv{a_{j, i}^{[l]}}{u_6} &= \pdv{a_{j, i}^{[l]}}{u_7} \pdv{u_7}{u_6} = u_4 = \exp(z_{j, i}^{[l]}) - \exp(-z_{j, i}^{[l]}), \\
\pdv{a_{j, i}^{[l]}}{u_5} &= \pdv{a_{j, i}^{[l]}}{u_6} \pdv{u_6}{u_5} = -u_4 \frac{1}{u_5^2} = -\frac{\exp(z_{j, i}^{[l]}) - \exp(-z_{j, i}^{[l]})}{(\exp(z_{j, i}^{[l]}) + \exp(-z_{j, i}^{[l]}))^2}, \\
\pdv{a_{j, i}^{[l]}}{u_4} &= \pdv{a_{j, i}^{[l]}}{u_7} \pdv{u_7}{u_4} = u_6 = \frac{1}{\exp(z_{j, i}^{[l]}) + \exp(-z_{j, i}^{[l]})}, \\
\pdv{a_{j, i}^{[l]}}{u_3} &= \pdv{a_{j, i}^{[l]}}{u_4} \pdv{u_4}{u_3} + \pdv{a_{j, i}^{[l]}}{u_5} \pdv{u_5}{u_3} \\
&= -u_6 - u_4 \frac{1}{u_5^2} \notag \\
&= -\frac{1}{\exp(z_{j, i}^{[l]}) + \exp(-z_{j, i}^{[l]})} - \frac{\exp(z_{j, i}^{[l]}) - \exp(-z_{j, i}^{[l]})}{(\exp(z_{j, i}^{[l]}) + \exp(-z_{j, i}^{[l]}))^2} \notag \\
&= -\frac{2 \exp(z_{j, i}^{[l]})}{(\exp(z_{j, i}^{[l]}) + \exp(-z_{j, i}^{[l]}))^2}, \notag \\
\pdv{a_{j, i}^{[l]}}{u_2} &= \pdv{a_{j, i}^{[l]}}{u_4} \pdv{u_4}{u_2} + \pdv{a_{j, i}^{[l]}}{u_5} \pdv{u_5}{u_2} \\
&= u_6 - u_4 \frac{1}{u_5^2} \notag \\
&= \frac{1}{\exp(z_{j, i}^{[l]}) + \exp(-z_{j, i}^{[l]})} - \frac{\exp(z_{j, i}^{[l]}) - \exp(-z_{j, i}^{[l]})}{(\exp(z_{j, i}^{[l]}) + \exp(-z_{j, i}^{[l]}))^2} \notag \\
&= \frac{2 \exp(-z_{j, i}^{[l]})}{(\exp(z_{j, i}^{[l]}) + \exp(-z_{j, i}^{[l]}))^2}, \notag \\
\pdv{a_{j, i}^{[l]}}{u_1} &= \pdv{a_{j, i}^{[l]}}{u_3} \pdv{u_3}{u_1} \\
&= \Bigl(-u_6 - u_4 \frac{1}{u_5^2}\Bigr) \exp(u_1) \notag \\
&= -\frac{2 \exp(z_{j, i}^{[l]}) \exp(-z_{j, i}^{[l]})}{(\exp(z_{j, i}^{[l]}) + \exp(-z_{j, i}^{[l]}))^2}, \notag \\
\pdv{a_{j, i}^{[l]}}{u_0} &= \pdv{a_{j, i}^{[l]}}{u_1} \pdv{u_1}{u_0} + \pdv{a_{j, i}^{[l]}}{u_2} \pdv{u_2}{u_0} \\
&= -\Bigl(-u_6 - u_4 \frac{1}{u_5^2}\Bigr) \exp(u_1) + \Bigl(u_6 - u_4 \frac{1}{u_5^2}\Bigr) \exp(u_0) \notag \\
&= \frac{2 \exp(z_{j, i}^{[l]}) \exp(-z_{j, i}^{[l]})}{(\exp(z_{j, i}^{[l]}) + \exp(-z_{j, i}^{[l]}))^2} + \frac{2 \exp(z_{j, i}^{[l]}) \exp(-z_{j, i}^{[l]})}{(\exp(z_{j, i}^{[l]}) + \exp(-z_{j, i}^{[l]}))^2} \notag \\
&= \frac{4 \exp(z_{j, i}^{[l]}) \exp(-z_{j, i}^{[l]})}{(\exp(z_{j, i}^{[l]}) + \exp(-z_{j, i}^{[l]}))^2}. \notag
\end{align*}
$$

It follows that

$$
\begin{equation*}
\begin{split}
\pdv{a_{j, i}^{[l]}}{z_{j, i}^{[l]}} &= \frac{4 \exp(z_{j, i}^{[l]}) \exp(-z_{j, i}^{[l]})}{(\exp(z_{j, i}^{[l]}) + \exp(-z_{j, i}^{[l]}))^2} \\
&= \frac{\exp(z_{j, i}^{[l]})^2 + 2 \exp(z_{j, i}^{[l]}) \exp(-z_{j, i}^{[l]}) + \exp(-z_{j, i}^{[l]})^2}{(\exp(z_{j, i}^{[l]}) + \exp(-z_{j, i}^{[l]}))^2} \\
&\peq \negmedspace {} - \frac{\exp(z_{j, i}^{[l]})^2 - 2 \exp(z_{j, i}^{[l]}) \exp(-z_{j, i}^{[l]}) + \exp(-z_{j, i}^{[l]})^2}{(\exp(z_{j, i}^{[l]}) + \exp(-z_{j, i}^{[l]}))^2} \\
&= 1 - \frac{(\exp(z_{j, i}^{[l]}) - \exp(-z_{j, i}^{[l]}))^2}{(\exp(z_{j, i}^{[l]}) + \exp(-z_{j, i}^{[l]}))^2} \\
&= 1 - a_{j, i}^{[l]} a_{j, i}^{[l]}.
\end{split}
\end{equation*}
$$

Similiar to the sigmoid activation function, we also have

$$
\begin{equation*}
\pdv{a_{\jj, i}^{[l]}}{z_{j, i}^{[l]}} = 0, \quad \forall \jj \ne j.
\end{equation*}
$$

Thus,

$$
\begin{equation*}
\begin{split}
\pdv{J}{z_{j, i}^{[l]}} &= \sum_\jj \pdv{J}{a_{\jj, i}^{[l]}} \pdv{a_{\jj, i}^{[l]}}{z_{j, i}^{[l]}} \\
&= \pdv{J}{a_{j, i}^{[l]}} \pdv{a_{j, i}^{[l]}}{z_{j, i}^{[l]}} + \sum_{\jj \ne j} \pdv{J}{a_{\jj, i}^{[l]}} \pdv{a_{\jj, i}^{[l]}}{z_{j, i}^{[l]}} \\
&= \pdv{J}{a_{j, i}^{[l]}} (1 - a_{j, i}^{[l]} a_{j, i}^{[l]}),
\end{split}
\end{equation*}
$$

which implies that

$$
\begin{equation}
\pdv{J}{\vec{Z}^{[l]}} = \pdv{J}{\vec{A}^{[l]}} \odot (1 - \vec{A}^{[l]} \odot \vec{A}^{[l]}).
\end{equation}
$$

## Softmax

The softmax activation function is given by

$$
\begin{equation*}
\begin{split}
a_{j, i}^{[l]} &= g_j^{[l]}(z_{1, i}^{[l]}, \dots, z_{j, i}^{[l]}, \dots, z_{n^{[l]}, i}^{[l]}) \\
&= \frac{\exp(z_{j, i}^{[l]})}{\sum_\jj \exp(z_{\jj, i}^{[l]})}.
\end{split}
\end{equation*}
$$

Vectorization results in

$$
\begin{equation}
\vec{A}^{[l]} = \frac{1}{\broadcast(\underbrace{\sum_{\text{axis} = 0} \exp(\vec{Z}^{[l]})}_\text{row vector})} \odot \exp(\vec{Z}^{[l]}).
\end{equation}
$$

To begin with, we construct a computation graph for the $$j$$th activation of the current layer:

$$
\begin{align*}
u_{-1} &= z_{j, i}^{[l]}, \\
u_{0, \jj} &= z_{\jj, i}^{[l]}, &&\forall \jj \ne j, \\
u_1 &= \exp(u_{-1}), \\
u_{2, \jj} &= \exp(u_{0, \jj}), &&\forall \jj \ne j, \\
u_3 &= u_1 + \sum_{\jj \ne j} u_{2, \jj}, \\
u_4 &= \frac{1}{u_3}, \\
u_5 &= u_1 u_4 = a_{j, i}^{[l]}.
\end{align*}
$$

By applying the chain rule, we get

$$
\begin{align*}
\pdv{a_{j, i}^{[l]}}{u_5} &= 1, \\
\pdv{a_{j, i}^{[l]}}{u_4} &= \pdv{a_{j, i}^{[l]}}{u_5} \pdv{u_5}{u_4} = u_1 = \exp(z_{j, i}^{[l]}), \\
\pdv{a_{j, i}^{[l]}}{u_3} &= \pdv{a_{j, i}^{[l]}}{u_4} \pdv{u_4}{u_3} = -u_1 \frac{1}{u_3^2} = -\frac{\exp(z_{j, i}^{[l]})}{(\sum_\jj \exp(z_{\jj, i}^{[l]}))^2}, \\
\pdv{a_{j, i}^{[l]}}{u_1} &= \pdv{a_{j, i}^{[l]}}{u_3} \pdv{u_3}{u_1} + \pdv{a_{j, i}^{[l]}}{u_5} \pdv{u_5}{u_1} \\
&= -u_1 \frac{1}{u_3^2} + u_4 \notag \\
&= -\frac{\exp(z_{j, i}^{[l]})}{(\sum_\jj \exp(z_{\jj, i}^{[l]}))^2} + \frac{1}{\sum_\jj \exp(z_{\jj, i}^{[l]})}, \notag \\
\pdv{a_{j, i}^{[l]}}{u_{-1}} &= \pdv{a_{j, i}^{[l]}}{u_1} \pdv{u_1}{u_{-1}} \\
&= \Bigl(-u_1 \frac{1}{u_3^2} + u_4\Bigr) \exp(u_{-1}) \notag \\
&= -\frac{\exp(z_{j, i}^{[l]})^2}{(\sum_\jj \exp(z_{\jj, i}^{[l]}))^2} + \frac{\exp(z_{j, i}^{[l]})}{\sum_\jj \exp(z_{\jj, i}^{[l]})}. \notag
\end{align*}
$$

Next, we need to take into account that $$z_{j, i}^{[l]}$$ also affects other activations in the same layer:

$$
\begin{align*}
u_{-1} &= z_{j, i}^{[l]}, \\
u_{0, \jj} &= z_{\jj, i}^{[l]}, &&\forall \jj \ne j, \\
u_1 &= \exp(u_{-1}), \\
u_{2, \jj} &= \exp(u_{0, \jj}), &&\forall \jj \ne j, \\
u_3 &= u_1 + \sum_{\jj \ne j} u_{2, \jj}, \\
u_4 &= \frac{1}{u_3}, \\
u_5 &= u_{2, \jj} u_4 = a_{\jj, i}^{[l]}, &&\forall \jj \ne j.
\end{align*}
$$

Backward propagation gives us the remaining partial derivatives:

$$
\begin{align*}
\pdv{a_{\jj, i}^{[l]}}{u_5} &= 1, \\
\pdv{a_{\jj, i}^{[l]}}{u_4} &= \pdv{a_{\jj, i}^{[l]}}{u_5} \pdv{u_5}{u_4} = u_{2, \jj} = \exp(z_{\jj, i}^{[l]}), \\
\pdv{a_{\jj, i}^{[l]}}{u_3} &= \pdv{a_{\jj, i}^{[l]}}{u_4} \pdv{u_4}{u_3} = -u_{2, \jj} \frac{1}{u_3^2} = -\frac{\exp(z_{\jj, i}^{[l]})}{(\sum_\jj \exp(z_{\jj, i}^{[l]}))^2}, \\
\pdv{a_{\jj, i}^{[l]}}{u_1} &= \pdv{a_{\jj, i}^{[l]}}{u_3} \pdv{u_3}{u_1} = -u_{2, \jj} \frac{1}{u_3^2} = -\frac{\exp(z_{\jj, i}^{[l]})}{(\sum_\jj \exp(z_{\jj, i}^{[l]}))^2}, \\
\pdv{a_{\jj, i}^{[l]}}{u_{-1}} &= \pdv{a_{\jj, i}^{[l]}}{u_1} \pdv{u_1}{u_{-1}} = -u_{2, \jj} \frac{1}{u_3^2} \exp(u_{-1}) = -\frac{\exp(z_{\jj, i}^{[l]}) \exp(z_{j, i}^{[l]})}{(\sum_\jj \exp(z_{\jj, i}^{[l]}))^2}.
\end{align*}
$$

We now know that

$$
\begin{align*}
\pdv{a_{j, i}^{[l]}}{z_{j, i}^{[l]}} &= -\frac{\exp(z_{j, i}^{[l]})^2}{(\sum_\jj \exp(z_{\jj, i}^{[l]}))^2} + \frac{\exp(z_{j, i}^{[l]})}{\sum_\jj \exp(z_{\jj, i}^{[l]})} \\
&= a_{j, i}^{[l]} (1 - a_{j, i}^{[l]}), \notag \\
\pdv{a_{\jj, i}^{[l]}}{z_{j, i}^{[l]}} &= -\frac{\exp(z_{\jj, i}^{[l]}) \exp(z_{j, i}^{[l]})}{(\sum_\jj \exp(z_{\jj, i}^{[l]}))^2} \\
&= -a_{\jj, i}^{[l]} a_{j, i}^{[l]}, \quad \forall \jj \ne j. \notag
\end{align*}
$$

Hence,

$$
\begin{equation*}
\begin{split}
\pdv{J}{z_{j, i}^{[l]}} &= \sum_\jj \pdv{J}{a_{\jj, i}^{[l]}} \pdv{a_{\jj, i}^{[l]}}{z_{j, i}^{[l]}} \\
&= \pdv{J}{a_{j, i}^{[l]}} \pdv{a_{j, i}^{[l]}}{z_{j, i}^{[l]}} + \sum_{\jj \ne j} \pdv{J}{a_{\jj, i}^{[l]}} \pdv{a_{\jj, i}^{[l]}}{z_{j, i}^{[l]}} \\
&= \pdv{J}{a_{j, i}^{[l]}} a_{j, i}^{[l]} (1 - a_{j, i}^{[l]}) - \sum_{\jj \ne j} \pdv{J}{a_{\jj, i}^{[l]}} a_{\jj, i}^{[l]} a_{j, i}^{[l]} \\
&= a_{j, i}^{[l]} \Bigl(\pdv{J}{a_{j, i}^{[l]}} (1 - a_{j, i}^{[l]}) - \sum_{\jj \ne j} \pdv{J}{a_{\jj, i}^{[l]}} a_{\jj, i}^{[l]}\Bigr) \\
&= a_{j, i}^{[l]} \Bigl(\pdv{J}{a_{j, i}^{[l]}} (1 - a_{j, i}^{[l]}) - \sum_\jj \pdv{J}{a_{\jj, i}^{[l]}} a_{\jj, i}^{[l]} + \pdv{J}{a_{j, i}^{[l]}} a_{j, i}^{[l]}\Bigr) \\
&= a_{j, i}^{[l]} \Bigl(\pdv{J}{a_{j, i}^{[l]}} - \sum_\jj \pdv{J}{a_{\jj, i}^{[l]}} a_{\jj, i}^{[l]}\Bigr),
\end{split}
\end{equation*}
$$

which we can vectorize as

{% raw %}
$$
\begin{equation*}
\pdv{J}{\vec{z}_{*, i}^{[l]}} = \vec{a}_{*, i}^{[l]} \odot \Bigl(\pdv{J}{\vec{a}_{*, i}^{[l]}} - \underbrace{{\vec{a}_{*, i}^{[l]}}^\T \pdv{J}{\vec{a}_{*, i}^{[l]}}}_{\text{scalar}}\Bigr).
\end{equation*}
$$
{% endraw %}

Let us not stop with the vectorization just yet:

$$
\begin{equation}
\pdv{J}{\vec{Z}^{[l]}} = \vec{A}^{[l]} \odot \Bigl(\pdv{J}{\vec{A}^{[l]}} - \broadcast\bigl(\underbrace{\sum_{\text{axis} = 0} \pdv{J}{\vec{A}^{[l]}} \odot \vec{A}^{[l]}}_\text{row vector}\bigr)\Bigr).
\end{equation}
$$
