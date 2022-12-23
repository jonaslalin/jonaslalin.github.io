---
title: "Feedforward Neural Networks in Depth, Part 1: Forward and Backward Propagations"
---

This post is the first of a three-part series in which we set out to derive the mathematics behind feedforward neural networks. They have

* an input and an output layer with at least one hidden layer in between,
* fully-connected layers, which means that each node in one layer connects to every node in the following layer, and
* ways to introduce nonlinearity by means of activation functions.

We start with forward propagation, which involves computing predictions and the associated cost of these predictions.

## Forward Propagation

Settling on what notations to use is tricky since we only have so many letters in the Roman alphabet. As you browse the Internet, you will likely find derivations that have used different notations than the ones we are about to introduce. However, and fortunately, there is no right or wrong here; it is just a matter of taste. In particular, the notations used in this series take inspiration from Andrew Ng's [Standard notations for Deep Learning]({% link /assets/deep-learning-notation.pdf %}){: target="_blank" }. If you make a comparison, you will find that we only change a couple of the details.

Now, whatever we come up with, we have to support

* multiple layers,
* several nodes in each layer,
* various activation functions,
* various types of cost functions, and
* mini-batches of training examples.

As a result, our definition of a node ends up introducing a fairly large number of notations:

$$
\begin{align}
z_{j, i}^{[l]} &= \sum_k w_{j, k}^{[l]} a_{k, i}^{[l - 1]} + b_j^{[l]}, \label{eq:z_scalar} \\
a_{j, i}^{[l]} &= g_j^{[l]}(z_{1, i}^{[l]}, \dots, z_{j, i}^{[l]}, \dots, z_{n^{[l]}, i}^{[l]}). \label{eq:a_scalar}
\end{align}
$$

Does the node definition look intimidating to you at first glance? Do not worry. Hopefully, it will make more sense once we have explained the notations, which we shall do next:

<div class="overflow-wrapper" markdown="1">
| Entity | Description |
| --- | --- |
| $$l$$ | The current layer $$l = 1, \dots, L$$, where $$L$$ is the number of layers that have weights and biases. We use $$l = 0$$ and $$l = L$$ to denote the input and output layers. |
| $$n^{[l]}$$ | The number of nodes in the current layer. |
| $$n^{[l - 1]}$$ | The number of nodes in the previous layer. |
| $$j$$ | The $$j$$th node of the current layer, $$j = 1, \dots, n^{[l]}$$. |
| $$k$$ | The $$k$$th node of the previous layer, $$k = 1, \dots, n^{[l - 1]}$$. |
| $$i$$ | The current training example $$i = 1, \dots, m$$, where $$m$$ is the number of training examples. |
| $$z_{j, i}^{[l]}$$ | A weighted sum of the activations of the previous layer, shifted by a bias. |
| $$w_{j, k}^{[l]}$$ | A weight that scales the $$k$$th activation of the previous layer. |
| $$b_j^{[l]}$$ | A bias in the current layer. |
| $$a_{j, i}^{[l]}$$ | An activation in the current layer. |
| $$a_{k, i}^{[l - 1]}$$ | An activation in the previous layer. |
| $$g_j^{[l]}$$ | An activation function $$g_j^{[l]} \colon \R^{n^{[l]}} \to \R$$ used in the current layer. |

</div>

To put it concisely, a node in the current layer depends on every node in the previous layer, and the following visualization can help us see that more clearly:

<figure class="overflow-wrapper">
  <svg id="nn-node-current-layer" class="nn" width="480" height="360" viewBox="240 0 480 360"></svg>
  <figcaption>Figure 1: A node in the current layer.</figcaption>
</figure>

Moreover, a node in the previous layer affects every node in the current layer, and with a change in highlighting, we will also be able to see that more clearly:

<figure class="overflow-wrapper">
  <svg id="nn-node-previous-layer" class="nn" width="480" height="360" viewBox="240 0 480 360"></svg>
  <figcaption>Figure 2: A node in the previous layer.</figcaption>
</figure>

In the future, we might want to write an implement from scratch in, for example, Python. To take advantage of the heavily optimized versions of vector and matrix operations that come bundled with libraries such as NumPy, we need to vectorize $$\eqref{eq:z_scalar}$$ and $$\eqref{eq:a_scalar}$$.

To begin with, we vectorize the nodes:

$$
\begin{align*}
\begin{bmatrix}
z_{1, i}^{[l]} \\
\vdots \\
z_{j, i}^{[l]} \\
\vdots \\
z_{n^{[l]}, i}^{[l]}
\end{bmatrix} &=
\begin{bmatrix}
w_{1, 1}^{[l]} & \dots & w_{1, k}^{[l]} & \dots & w_{1, n^{[l - 1]}}^{[l]} \\
\vdots & \ddots & \vdots & \ddots & \vdots \\
w_{j, 1}^{[l]} & \dots & w_{j, k}^{[l]} & \dots & w_{j, n^{[l - 1]}}^{[l]} \\
\vdots & \ddots & \vdots & \ddots & \vdots \\
w_{n^{[l]}, 1}^{[l]} & \dots & w_{n^{[l]}, k}^{[l]} & \dots & w_{n^{[l]}, n^{[l - 1]}}^{[l]}
\end{bmatrix}
\begin{bmatrix}
a_{1, i}^{[l - 1]} \\
\vdots \\
a_{k, i}^{[l - 1]} \\
\vdots \\
a_{n^{[l - 1]}, i}^{[l - 1]}
\end{bmatrix} +
\begin{bmatrix}
b_1^{[l]} \\
\vdots \\
b_j^{[l]} \\
\vdots \\
b_{n^{[l]}}^{[l]}
\end{bmatrix}, \\
\begin{bmatrix}
a_{1, i}^{[l]} \\
\vdots \\
a_{j, i}^{[l]} \\
\vdots \\
a_{n^{[l]}, i}^{[l]}
\end{bmatrix} &=
\begin{bmatrix}
g_1^{[l]}(z_{1, i}^{[l]}, \dots, z_{j, i}^{[l]}, \dots, z_{n^{[l]}, i}^{[l]}) \\
\vdots \\
g_j^{[l]}(z_{1, i}^{[l]}, \dots, z_{j, i}^{[l]}, \dots, z_{n^{[l]}, i}^{[l]}) \\
\vdots \\
g_{n^{[l]}}^{[l]}(z_{1, i}^{[l]}, \dots, z_{j, i}^{[l]}, \dots, z_{n^{[l]}, i}^{[l]}) \\
\end{bmatrix},
\end{align*}
$$

which we can write as

$$
\begin{align}
\vec{z}_{:, i}^{[l]} &= \vec{W}^{[l]} \vec{a}_{:, i}^{[l - 1]} + \vec{b}^{[l]}, \label{eq:z} \\
\vec{a}_{:, i}^{[l]} &= \vec{g}^{[l]}(\vec{z}_{:, i}^{[l]}), \label{eq:a}
\end{align}
$$

where $$\vec{z}_{:, i}^{[l]} \in \R^{n^{[l]}}$$, $$\vec{W}^{[l]} \in \R^{n^{[l]} \times n^{[l - 1]}}$$, $$\vec{b}^{[l]} \in \R^{n^{[l]}}$$, $$\vec{a}_{:, i}^{[l]} \in \R^{n^{[l]}}$$, $$\vec{a}_{:, i}^{[l - 1]} \in \R^{n^{[l - 1]}}$$, and lastly, $$\vec{g}^{[l]} \colon \R^{n^{[l]}} \to \R^{n^{[l]}}$$. We have used a colon to clarify that $$\vec{z}_{:, i}^{[l]}$$ is the $$i$$th column of $$\vec{Z}^{[l]}$$, and so on.

Next, we vectorize the training examples:

$$
\begin{align}
\vec{Z}^{[l]} &=
\begin{bmatrix}
\vec{z}_{:, 1}^{[l]} & \dots & \vec{z}_{:, i}^{[l]} & \dots & \vec{z}_{:, m}^{[l]}
\end{bmatrix} \label{eq:Z} \\
&= \vec{W}^{[l]}
\begin{bmatrix}
\vec{a}_{:, 1}^{[l - 1]} & \dots & \vec{a}_{:, i}^{[l - 1]} & \dots & \vec{a}_{:, m}^{[l - 1]}
\end{bmatrix} +
\begin{bmatrix}
\vec{b}^{[l]} & \dots & \vec{b}^{[l]} & \dots & \vec{b}^{[l]}
\end{bmatrix} \notag \\
&= \vec{W}^{[l]} \vec{A}^{[l - 1]} + \broadcast(\vec{b}^{[l]}), \notag \\
\vec{A}^{[l]} &=
\begin{bmatrix}
\vec{a}_{:, 1}^{[l]} & \dots & \vec{a}_{:, i}^{[l]} & \dots & \vec{a}_{:, m}^{[l]}
\end{bmatrix}, \label{eq:A}
\end{align}
$$

where $$\vec{Z}^{[l]} \in \R^{n^{[l]} \times m}$$, $$\vec{A}^{[l]} \in \R^{n^{[l]} \times m}$$, and $$\vec{A}^{[l - 1]} \in \R^{n^{[l - 1]} \times m}$$. In addition, have a look at [the NumPy documentation](https://numpy.org/doc/stable/user/basics.broadcasting.html){: target="_blank" } if you want to read a well-written explanation of broadcasting.

We would also like to establish two additional notations:

$$
\begin{align}
\vec{A}^{[0]} &= \vec{X}, \label{eq:A_zero} \\
\vec{A}^{[L]} &= \vec{\hat{Y}}, \label{eq:A_L}
\end{align}
$$

where $$\vec{X} \in \R^{n^{[0]} \times m}$$ denotes the inputs and $$\vec{\hat{Y}} \in \R^{n^{[L]} \times m}$$ denotes the predictions/outputs.

Finally, we are ready to define the cost function:

$$
\begin{equation}
J = f(\vec{\hat{Y}}, \vec{Y}) = f(\vec{A}^{[L]}, \vec{Y}), \label{eq:J}
\end{equation}
$$

where $$\vec{Y} \in \R^{n^{[L]} \times m}$$ denotes the targets and $$f \colon \R^{2 n^{[L]}} \to \R$$ can be tailored to our needs.

We are done with forward propagation! Next up: backward propagation, also known as backpropagation, which involves computing the gradient of the cost function with respect to the weights and biases.

## Backward Propagation

We will make heavy use of the chain rule in this section, and to understand better how it works, we first apply the chain rule to the following example:

$$
\begin{align}
u_i &= g_i(x_1, \dots, x_j, \dots, x_n), \label{eq:example_u_scalar} \\
y_k &= f_k(u_1, \dots, u_i, \dots, u_m). \label{eq:example_y_scalar}
\end{align}
$$

Note that $$x_j$$ may affect $$u_1, \dots, u_i, \dots, u_m$$, and $$y_k$$ may depend on $$u_1, \dots, u_i, \dots, u_m$$; thus,

$$
\begin{equation}
\pdv{y_k}{x_j} = \sum_i \pdv{y_k}{u_i} \pdv{u_i}{x_j}. \label{eq:chain_rule}
\end{equation}
$$

Great! If we ever get stuck trying to compute or understand some partial derivative, we can always go back to $$\eqref{eq:example_u_scalar}$$, $$\eqref{eq:example_y_scalar}$$, and $$\eqref{eq:chain_rule}$$. Hopefully, these equations will provide the clues necessary to move forward. However, be extra careful not to confuse the notation used for the chain rule example with the notation we use elsewhere in this series. The overlap is unintentional.

Now, let us concentrate on the task at hand:

$$
\begin{align}
\pdv{J}{w_{j, k}^{[l]}} &= \sum_i \pdv{J}{z_{j, i}^{[l]}} \pdv{z_{j, i}^{[l]}}{w_{j, k}^{[l]}} = \sum_i \pdv{J}{z_{j, i}^{[l]}} a_{k, i}^{[l - 1]}, \label{eq:dw_scalar} \\
\pdv{J}{b_j^{[l]}} &= \sum_i \pdv{J}{z_{j, i}^{[l]}} \pdv{z_{j, i}^{[l]}}{b_j^{[l]}} = \sum_i \pdv{J}{z_{j, i}^{[l]}}. \label{eq:db_scalar}
\end{align}
$$

Vectorization results in

$$
\begin{align*}
&
\begin{bmatrix}
\dpdv{J}{w_{1, 1}^{[l]}} & \dots & \dpdv{J}{w_{1, k}^{[l]}} & \dots & \dpdv{J}{w_{1, n^{[l - 1]}}^{[l]}} \\
\vdots & \ddots & \vdots & \ddots & \vdots \\
\dpdv{J}{w_{j, 1}^{[l]}} & \dots & \dpdv{J}{w_{j, k}^{[l]}} & \dots & \dpdv{J}{w_{j, n^{[l - 1]}}^{[l]}} \\
\vdots & \ddots & \vdots & \ddots & \vdots \\
\dpdv{J}{w_{n^{[l]}, 1}^{[l]}} & \dots & \dpdv{J}{w_{n^{[l]}, k}^{[l]}} & \dots & \dpdv{J}{w_{n^{[l]}, n^{[l - 1]}}^{[l]}}
\end{bmatrix} \\
&=
\begin{bmatrix}
\dpdv{J}{z_{1, 1}^{[l]}} & \dots & \dpdv{J}{z_{1, i}^{[l]}} & \dots & \dpdv{J}{z_{1, m}^{[l]}} \\
\vdots & \ddots & \vdots & \ddots & \vdots \\
\dpdv{J}{z_{j, 1}^{[l]}} & \dots & \dpdv{J}{z_{j, i}^{[l]}} & \dots & \dpdv{J}{z_{j, m}^{[l]}} \\
\vdots & \ddots & \vdots & \ddots & \vdots \\
\dpdv{J}{z_{n^{[l]}, 1}^{[l]}} & \dots & \dpdv{J}{z_{n^{[l]}, i}^{[l]}} & \dots & \dpdv{J}{z_{n^{[l]}, m}^{[l]}}
\end{bmatrix} \notag \\
&\peq {} \cdot
\begin{bmatrix}
a_{1, 1}^{[l - 1]} & \dots & a_{k, 1}^{[l - 1]} & \dots & a_{n^{[l - 1]}, 1}^{[l - 1]} \\
\vdots & \ddots & \vdots & \ddots & \vdots \\
a_{1, i}^{[l - 1]} & \dots & a_{k, i}^{[l - 1]} & \dots & a_{n^{[l - 1]}, i}^{[l - 1]} \\
\vdots & \ddots & \vdots & \ddots & \vdots \\
a_{1, m}^{[l - 1]} & \dots & a_{k, m}^{[l - 1]} & \dots & a_{n^{[l - 1]}, m}^{[l - 1]}
\end{bmatrix}, \notag \\
&
\begin{bmatrix}
\dpdv{J}{b_1^{[l]}} \\
\vdots \\
\dpdv{J}{b_j^{[l]}} \\
\vdots \\
\dpdv{J}{b_{n^{[l]}}^{[l]}}
\end{bmatrix} =
\begin{bmatrix}
\dpdv{J}{z_{1, 1}^{[l]}} \\
\vdots \\
\dpdv{J}{z_{j, 1}^{[l]}} \\
\vdots \\
\dpdv{J}{z_{n^{[l]}, 1}^{[l]}}
\end{bmatrix} + \dots +
\begin{bmatrix}
\dpdv{J}{z_{1, i}^{[l]}} \\
\vdots \\
\dpdv{J}{z_{j, i}^{[l]}} \\
\vdots \\
\dpdv{J}{z_{n^{[l]}, i}^{[l]}}
\end{bmatrix} + \dots +
\begin{bmatrix}
\dpdv{J}{z_{1, m}^{[l]}} \\
\vdots \\
\dpdv{J}{z_{j, m}^{[l]}} \\
\vdots \\
\dpdv{J}{z_{n^{[l]}, m}^{[l]}}
\end{bmatrix},
\end{align*}
$$

which we can write as

$$
\begin{align}
\pdv{J}{\vec{W}^{[l]}} &= \sum_i \pdv{J}{\vec{z}_{:, i}^{[l]}} {\vec{a}_{:, i}^{[l - 1]}}^\T = \pdv{J}{\vec{Z}^{[l]}} {\vec{A}^{[l - 1]}}^\T, \label{eq:dW} \\
\pdv{J}{\vec{b}^{[l]}} &= \sum_i \pdv{J}{\vec{z}_{:, i}^{[l]}} = \underbrace{\sum_{\text{axis} = 1} \pdv{J}{\vec{Z}^{[l]}}}_\text{column vector}, \label{eq:db}
\end{align}
$$

where $$\pdv{J}{\vec{z}_{:, i}^{[l]}} \in \R^{n^{[l]}}$$, $$\pdv{J}{\vec{Z}^{[l]}} \in \R^{n^{[l]} \times m}$$, $$\pdv{J}{\vec{W}^{[l]}} \in \R^{n^{[l]} \times n^{[l - 1]}}$$, and $$\pdv{J}{\vec{b}^{[l]}} \in \R^{n^{[l]}}$$.

Looking back at $$\eqref{eq:dw_scalar}$$ and $$\eqref{eq:db_scalar}$$, we see that the only unknown entity is $$\pdv{J}{z_{j, i}^{[l]}}$$. By applying the chain rule once again, we get

$$
\begin{equation}
\pdv{J}{z_{j, i}^{[l]}} = \sum_p \pdv{J}{a_{p, i}^{[l]}} \pdv{a_{p, i}^{[l]}}{z_{j, i}^{[l]}}, \label{eq:dz_scalar}
\end{equation}
$$

where $$p = 1, \dots, n^{[l]}$$.

Next, we present the vectorized version:

$$
\begin{equation*}
\begin{bmatrix}
\dpdv{J}{z_{1, i}^{[l]}} \\
\vdots \\
\dpdv{J}{z_{j, i}^{[l]}} \\
\vdots \\
\dpdv{J}{z_{n^{[l]}, i}^{[l]}}
\end{bmatrix} =
\begin{bmatrix}
\dpdv{a_{1, i}^{[l]}}{z_{1, i}^{[l]}} & \dots & \dpdv{a_{j, i}^{[l]}}{z_{1, i}^{[l]}} & \dots & \dpdv{a_{n^{[l]}, i}^{[l]}}{z_{1, i}^{[l]}} \\
\vdots & \ddots & \vdots & \ddots & \vdots \\
\dpdv{a_{1, i}^{[l]}}{z_{j, i}^{[l]}} & \dots & \dpdv{a_{j, i}^{[l]}}{z_{j, i}^{[l]}} & \dots & \dpdv{a_{n^{[l]}, i}^{[l]}}{z_{j, i}^{[l]}} \\
\vdots & \ddots & \vdots & \ddots & \vdots \\
\dpdv{a_{1, i}^{[l]}}{z_{n^{[l]}, i}^{[l]}} & \dots & \dpdv{a_{j, i}^{[l]}}{z_{n^{[l]}, i}^{[l]}} & \dots & \dpdv{a_{n^{[l]}, i}^{[l]}}{z_{n^{[l]}, i}^{[l]}}
\end{bmatrix}
\begin{bmatrix}
\dpdv{J}{a_{1, i}^{[l]}} \\
\vdots \\
\dpdv{J}{a_{j, i}^{[l]}} \\
\vdots \\
\dpdv{J}{a_{n^{[l]}, i}^{[l]}}
\end{bmatrix},
\end{equation*}
$$

which compresses into

$$
\begin{equation}
\pdv{J}{\vec{z}_{:, i}^{[l]}} = \pdv{\vec{a}_{:, i}^{[l]}}{\vec{z}_{:, i}^{[l]}} \pdv{J}{\vec{a}_{:, i}^{[l]}}, \label{eq:dz}
\end{equation}
$$

where $$\pdv{J}{\vec{a}_{:, i}^{[l]}} \in \R^{n^{[l]}}$$ and $$\pdv{\vec{a}_{:, i}^{[l]}}{\vec{z}_{:, i}^{[l]}} \in \R^{n^{[l]} \times n^{[l]}}$$.

We have already encountered

$$
\begin{equation}
\pdv{J}{\vec{Z}^{[l]}} =
\begin{bmatrix}
\dpdv{J}{\vec{z}_{:, 1}^{[l]}} & \dots & \dpdv{J}{\vec{z}_{:, i}^{[l]}} & \dots & \dpdv{J}{\vec{z}_{:, m}^{[l]}}
\end{bmatrix}, \label{eq:dZ}
\end{equation}
$$

and for the sake of completeness, we also clarify that

$$
\begin{equation}
\pdv{J}{\vec{A}^{[l]}} =
\begin{bmatrix}
\dpdv{J}{\vec{a}_{:, 1}^{[l]}} & \dots & \dpdv{J}{\vec{a}_{:, i}^{[l]}} & \dots & \dpdv{J}{\vec{a}_{:, m}^{[l]}}
\end{bmatrix}, \label{eq:dA}
\end{equation}
$$

where $$\pdv{J}{\vec{A}^{[l]}} \in \R^{n^{[l]} \times m}$$.

On purpose, we have omitted the details of $$g_j^{[l]}(z_{1, i}^{[l]}, \dots, z_{j, i}^{[l]}, \dots, z_{n^{[l]}, i}^{[l]})$$; consequently, we cannot derive an analytic expression for $$\pdv{a_{j, i}^{[l]}}{z_{j, i}^{[l]}}$$, which we depend on in $$\eqref{eq:dz_scalar}$$. However, since [the second post]({% post_url 2021-12-21-feedforward-neural-networks-part-2 %}){: target="_blank" } of this series will be dedicated to activation functions, we will instead derive $$\pdv{a_{j, i}^{[l]}}{z_{j, i}^{[l]}}$$ there.

Furthermore, according to $$\eqref{eq:dz_scalar}$$, we see that $$\pdv{J}{z_{j, i}^{[l]}}$$ also depends on $$\pdv{J}{a_{j, i}^{[l]}}$$. Now, it might come as a surprise, but $$\pdv{J}{a_{j, i}^{[l]}}$$ has already been computed when we reach the $$l$$th layer during backward propagation. How did that happen, you may ask. The answer is that every layer paves the way for the previous layer by also computing $$\pdv{J}{a_{k, i}^{[l - 1]}}$$, which we shall do now:

$$
\begin{equation}
\pdv{J}{a_{k, i}^{[l - 1]}} = \sum_j \pdv{J}{z_{j, i}^{[l]}} \pdv{z_{j, i}^{[l]}}{a_{k, i}^{[l - 1]}} = \sum_j \pdv{J}{z_{j, i}^{[l]}} w_{j, k}^{[l]}. \label{eq:da_prev_scalar}
\end{equation}
$$

As usual, our next step is vectorization:

$$
\begin{equation*}
\begin{split}
&
\begin{bmatrix}
\dpdv{J}{a_{1, 1}^{[l - 1]}} & \dots & \dpdv{J}{a_{1, i}^{[l - 1]}} & \dots & \dpdv{J}{a_{1, m}^{[l - 1]}} \\
\vdots & \ddots & \vdots & \ddots & \vdots \\
\dpdv{J}{a_{k, 1}^{[l - 1]}} & \dots & \dpdv{J}{a_{k, i}^{[l - 1]}} & \dots & \dpdv{J}{a_{k, m}^{[l - 1]}} \\
\vdots & \ddots & \vdots & \ddots & \vdots \\
\dpdv{J}{a_{n^{[l - 1]}, 1}^{[l - 1]}} & \dots & \dpdv{J}{a_{n^{[l - 1]}, i}^{[l - 1]}} & \dots & \dpdv{J}{a_{n^{[l - 1]}, m}^{[l - 1]}}
\end{bmatrix} \\
&=
\begin{bmatrix}
w_{1, 1}^{[l]} & \dots & w_{j, 1}^{[l]} & \dots & w_{n^{[l]}, 1}^{[l]} \\
\vdots & \ddots & \vdots & \ddots & \vdots \\
w_{1, k}^{[l]} & \dots & w_{j, k}^{[l]} & \dots & w_{n^{[l]}, k}^{[l]} \\
\vdots & \ddots & \vdots & \ddots & \vdots \\
w_{1, n^{[l - 1]}}^{[l]} & \dots & w_{j, n^{[l - 1]}}^{[l]} & \dots & w_{n^{[l]}, n^{[l - 1]}}^{[l]}
\end{bmatrix} \\
&\peq {} \cdot
\begin{bmatrix}
\dpdv{J}{z_{1, 1}^{[l]}} & \dots & \dpdv{J}{z_{1, i}^{[l]}} & \dots & \dpdv{J}{z_{1, m}^{[l]}} \\
\vdots & \ddots & \vdots & \ddots & \vdots \\
\dpdv{J}{z_{j, 1}^{[l]}} & \dots & \dpdv{J}{z_{j, i}^{[l]}} & \dots & \dpdv{J}{z_{j, m}^{[l]}} \\
\vdots & \ddots & \vdots & \ddots & \vdots \\
\dpdv{J}{z_{n^{[l]}, 1}^{[l]}} & \dots & \dpdv{J}{z_{n^{[l]}, i}^{[l]}} & \dots & \dpdv{J}{z_{n^{[l]}, m}^{[l]}}
\end{bmatrix},
\end{split}
\end{equation*}
$$

which we can write as

$$
\begin{equation}
\pdv{J}{\vec{A}^{[l - 1]}} = {\vec{W}^{[l]}}^\T \pdv{J}{\vec{Z}^{[l]}}, \label{eq:dA_prev}
\end{equation}
$$

where $$\pdv{J}{\vec{A}^{[l - 1]}} \in \R^{n^{[l - 1]} \times m}$$.

## Summary

Forward propagation is seeded with $$\vec{A}^{[0]} = \vec{X}$$ and evaluates a set of recurrence relations to compute the predictions $$\vec{A}^{[L]} = {\vec{\hat{Y}}}$$. We also compute the cost $$J = f(\vec{\hat{Y}}, \vec{Y}) = f(\vec{A}^{[L]}, \vec{Y})$$.

Backward propagation, on the other hand, is seeded with $$\pdv{J}{\vec{A}^{[L]}} = \pdv{J}{\vec{\hat{Y}}}$$ and evaluates a different set of recurrence relations to compute $$\pdv{J}{\vec{W}^{[l]}}$$ and $$\pdv{J}{\vec{b}^{[l]}}$$. If not stopped prematurely, it eventually computes $$\pdv{J}{\vec{A}^{[0]}} = \pdv{J}{\vec{X}}$$, a partial derivative we usually ignore.

Moreover, let us visualize the inputs we use and the outputs we produce during the forward and backward propagations:

<figure class="overflow-wrapper">
  <svg id="bld-forward-propagation" class="bld" width="480" height="360"></svg>
  <svg id="bld-backward-propagation" class="bld" width="480" height="255"></svg>
  <figcaption>Figure 3: An overview of inputs and outputs.</figcaption>
</figure>

Now, you might have noticed that we have yet to derive an analytic expression for the backpropagation seed $$\pdv{J}{\vec{A}^{[L]}} = \pdv{J}{\vec{\hat{Y}}}$$. To recap, we have deferred the derivations that concern activation functions to [the second post]({% post_url 2021-12-21-feedforward-neural-networks-part-2 %}){: target="_blank" } of this series. Similarly, since [the third post]({% post_url 2021-12-22-feedforward-neural-networks-part-3 %}){: target="_blank" } will be dedicated to cost functions, we will instead address the derivation of the backpropagation seed there.

Last but not least: congratulations! You have made it to the end (of the first post). 🏅

{%- include d3.html -%}
{%- include neural-network-svg.html -%}
{%- include box-line-diagram-svg.html -%}

<link rel="stylesheet" href="{{ "/assets/css/neural-network.css" | relative_url }}">
<link rel="stylesheet" href="{{ "/assets/css/box-line-diagram.css" | relative_url }}">

<script>
  var nnThreeToThree =
    nn.buildNeuralNetworkWithCoordinates(
      [3, 3, 3, 3],  // nNodesPerLayer
      960,           // width
      360,           // height
      30             // nodeRadius
    );

  nn.drawNeuralNetwork(
    'nn-node-current-layer',  // svgId
    'nn',                     // cssPrefix
    nnThreeToThree            // neuralNetworkWithCoordinates
  );

  nn.drawNeuralNetwork(
    'nn-node-previous-layer',  // svgId
    'nn',                      // cssPrefix
    nnThreeToThree             // neuralNetworkWithCoordinates
  );

  var nnTextOptions = new nn.TextOptions(
    108,  // width
    30,   // height
    0.5   // position
  );

  var nnNodeTexts = [
    new nn.NodeText(1, 0, 'a_{k - 2, i}^[l - 1]'),
    new nn.NodeText(1, 1, 'a_{k - 1, i}^[l - 1]'),
    new nn.NodeText(1, 2, 'a_{k, i}^[l - 1]'),
    new nn.NodeText(2, 0, 'a_{j, i}^[l]'),
    new nn.NodeText(2, 1, 'a_{j + 1, i}^[l]'),
    new nn.NodeText(2, 2, 'a_{j + 2, i}^[l]')
  ];

  nn.annotateNeuralNetwork(
    'nn-node-current-layer',  // svgId
    'nn',                     // cssPrefix
    nnThreeToThree,           // neuralNetworkWithCoordinates
    nnNodeTexts,              // nodeTexts
    // linkTexts
    [
      new nn.LinkText(1, 0, 2, 0, 'w_{j, k - 2}^[l]'),
      new nn.LinkText(1, 1, 2, 0, 'w_{j, k - 1}^[l]'),
      new nn.LinkText(1, 2, 2, 0, 'w_{j, k}^[l]')
    ],
    nnTextOptions  // textOptions
  );

  nn.annotateNeuralNetwork(
    'nn-node-previous-layer',  // svgId
    'nn',                      // cssPrefix
    nnThreeToThree,            // neuralNetworkWithCoordinates
    nnNodeTexts,               // nodeTexts
    // linkTexts
    [
      new nn.LinkText(1, 2, 2, 0, 'w_{j, k}^[l]'),
      new nn.LinkText(1, 2, 2, 1, 'w_{j + 1, k}^[l]'),
      new nn.LinkText(1, 2, 2, 2, 'w_{j + 2, k}^[l]')
    ],
    nnTextOptions  // textOptions
  );
</script>

<script>
  var bldForwardPropagation =
    bld.buildBoxLineDiagramWithCoordinates(
      new bld.BoxLineDiagram(
        new bld.Box(
          200,  // width
          150,  // height
          50,   // padding
          // texts
          [
            'Z^[l]'
          ]
        ),
        [
          new bld.Line(bld.PLACEMENT_LEFT  , bld.ARROWHEAD_END, 'A^[l - 1]'),
          new bld.Line(bld.PLACEMENT_RIGHT , bld.ARROWHEAD_END, 'A^[l]'),
          new bld.Line(bld.PLACEMENT_TOP   , bld.ARROWHEAD_END, 'W^[l]'),
          new bld.Line(bld.PLACEMENT_TOP   , bld.ARROWHEAD_END, 'b^[l]'),
          new bld.Line(bld.PLACEMENT_BOTTOM, bld.ARROWHEAD_END, 'cache^[l]')
        ]
      ),
      480,  // width
      360   // height
    );

  var bldBackwardPropagation =
    bld.buildBoxLineDiagramWithCoordinates(
      new bld.BoxLineDiagram(
        new bld.Box(
          200,  // width
          150,  // height
          50,   // padding
          // texts
          [
            'dZ^[l]'
          ]
        ),
        [
          new bld.Line(bld.PLACEMENT_LEFT  , bld.ARROWHEAD_START, 'dA^[l - 1]'),
          new bld.Line(bld.PLACEMENT_RIGHT , bld.ARROWHEAD_START, 'dA^[l]'),
          new bld.Line(bld.PLACEMENT_BOTTOM, bld.ARROWHEAD_END  , 'dW^[l]'),
          new bld.Line(bld.PLACEMENT_BOTTOM, bld.ARROWHEAD_END  , 'db^[l]')
        ]
      ),
      480,  // width
      255   // height
    );

  var bldTextOptions = new bld.TextOptions(
    69,  // width
    30   // height
  );

  var bldArrowheadOptions = new bld.ArrowheadOptions(
    10,        // width
    10,        // height
    '#3f3f3f'  // fill
  );

  bld.drawBoxLineDiagram(
    'bld-forward-propagation',  // svgId
    'bld',                      // cssPrefix
    bldForwardPropagation,      // boxLineDiagramWithCoordinates
    bldTextOptions,             // textOptions
    bldArrowheadOptions         // arrowheadOptions
  );

  bld.drawBoxLineDiagram(
    'bld-backward-propagation',  // svgId
    'bld',                       // cssPrefix
    bldBackwardPropagation,      // boxLineDiagramWithCoordinates
    bldTextOptions,              // textOptions
    bldArrowheadOptions          // arrowheadOptions
  );
</script>
