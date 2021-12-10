---
title: How Backpropagation Is Able To Reduce the Time Spent on Computing Gradients
---

Backpropagation was originally introduced in the 1970s, but its importance was not fully appreciated until [Learning representations by back-propagating errors](https://www.nature.com/articles/323533a0){: target="_blank" } was published in 1986. With backpropagation, it became possible to use neural networks to solve problems that had previously been insoluble. Today, backpropagation is the workhorse of learning in neural networks. Without it, we would waste both time and energy. So how is backpropagation able to reduce the time spent on computing gradients? It all boils down to the computational complexity between applying the chain rule in forward versus reverse accumulation mode.

## Forward and Reverse Accumulation Modes

Suppose we have a function

$$
\begin{equation*}
y = f(g(h(x))).
\end{equation*}
$$

Let us decompose the function with the help of intermediate variables:

$$
\begin{align*}
u_0 &= x, \\
u_1 &= h(u_0), \\
u_2 &= g(u_1), \\
u_3 &= f(u_2) = y.
\end{align*}
$$

To compute the derivative $$\dv{y}{x}$$, we can traverse the chain rule

1. inside-out, or
2. outside-in.

We start with the inside-out traversal of the chain rule, i.e., the forward accumulation mode:

$$
\begin{align*}
\dv{u_0}{x} &= 1, \\
\dv{u_1}{x} &= \dv{u_1}{u_0} \dv{u_0}{x} = \dv{h(u_0)}{u_0}, \\
\dv{u_2}{x} &= \dv{u_2}{u_1} \dv{u_1}{x} = \dv{g(u_1)}{u_1} \dv{h(u_0)}{u_0}, \\
\dv{u_3}{x} &= \dv{u_3}{u_2} \dv{u_2}{x} = \dv{f(u_2)}{u_2} \dv{g(u_1)}{u_1} \dv{h(u_0)}{u_0}.
\end{align*}
$$

By contrast, the reverse accumulation mode performs the outside-in traversal of the chain rule, which more commonly is referred to as backpropagation:

$$
\begin{align*}
\dv{y}{u_3} &= 1, \\
\dv{y}{u_2} &= \dv{y}{u_3} \dv{u_3}{u_2} = \dv{f(u_2)}{u_2}, \\
\dv{y}{u_1} &= \dv{y}{u_2} \dv{u_2}{u_1} = \dv{f(u_2)}{u_2} \dv{g(u_1)}{u_1}, \\
\dv{y}{u_0} &= \dv{y}{u_1} \dv{u_1}{u_0} = \dv{f(u_2)}{u_2} \dv{g(u_1)}{u_1} \dv{h(u_0)}{u_0}.
\end{align*}
$$

Both methods reach

$$
\begin{equation*}
\dv{y}{x} = \dv{u_3}{x} = \dv{y}{u_0} = \dv{f(u_2)}{u_2} \dv{g(u_1)}{u_1} \dv{h(u_0)}{u_0},
\end{equation*}
$$

using the same number of computations; however, this is not always the case, as we soon will find out.

Note that the forward accumulation mode computes the recurrence relation

$$
\begin{equation*}
\dv{u_i}{x} = \dv{u_i}{u_{i - 1}} \dv{u_{i - 1}}{x}.
\end{equation*}
$$

In contrast, the reverse accumulation mode computes the recurrence relation

$$
\begin{equation*}
\dv{y}{u_i} = \dv{y}{u_{i + 1}} \dv{u_{i + 1}}{u_i}.
\end{equation*}
$$

Now, let us move on to a function $$f \colon \R^3 \to \R^2$$, where it will be easier to analyze the computational complexity of the forward and reverse accumulation modes.

## Example

To make a good comparison, we need an example with a different number of dependent variables than independent variables. The following function fulfills that requirement:

$$
\begin{align*}
y_1 &= x_1 (x_2 - x_3), \\
y_2 &= x_3 \log(1 - x_1).
\end{align*}
$$

Next, to make gradient computations as simple as possible, after decomposition, we make sure we are left with only basic arithmetic operations and elementary functions:

$$
\begin{align*}
u_{-2} &= x_1, \\
u_{-1} &= x_2, \\
u_0 &= x_3, \\
u_1 &= u_{-1} - u_0, \\
u_2 &= 1 - u_{-2}, \\
u_3 &= \log(u_2), \\
u_4 &= u_{-2} u_1 = y_1, \\
u_5 &= u_0 u_3 = y_2.
\end{align*}
$$

Now, we are ready to compute the partial derivatives $$\pdv{y_1}{x_1}$$, $$\pdv{y_1}{x_2}$$, $$\pdv{y_1}{x_3}$$, $$\pdv{y_2}{x_1}$$, $$\pdv{y_2}{x_2}$$, and $$\pdv{y_2}{x_3}$$. Once again, we start with the inside-out traversal of the chain rule.

### The Forward Accumulation Mode

__Iteration 1:__

$$
\begin{align*}
\pdv{u_{-2}}{x_1} &= 1, \\
\pdv{u_{-1}}{x_1} &= 0, \\
\pdv{u_0}{x_1} &= 0, \\
\pdv{u_1}{x_1} &= \pdv{u_1}{u_{-1}} \pdv{u_{-1}}{x_1} + \pdv{u_1}{u_0} \pdv{u_0}{x_1} = 0, \\
\pdv{u_2}{x_1} &= \pdv{u_2}{u_{-2}} \pdv{u_{-2}}{x_1} = -1, \\
\pdv{u_3}{x_1} &= \pdv{u_3}{u_2} \pdv{u_2}{x_1} = -\frac{1}{u_2} = -\frac{1}{1 - x_1}, \\
\pdv{u_4}{x_1} &= \pdv{u_4}{u_{-2}} \pdv{u_{-2}}{x_1} + \pdv{u_4}{u_1} \pdv{u_1}{x_1} = u_1 = x_2 - x_3, \\
\pdv{u_5}{x_1} &= \pdv{u_5}{u_0} \pdv{u_0}{x_1} + \pdv{u_5}{u_3} \pdv{u_3}{x_1} = -u_0 \frac{1}{u_2} = -\frac{x_3}{1 - x_1}.
\end{align*}
$$

Computing the partial derivative of every intermediate variable once gives us $$\pdv{y_1}{x_1} = x_2 - x_3$$ and $$\pdv{y_2}{x_1} = -x_3 / (1 - x_1)$$.

__Iteration 2:__

$$
\begin{align*}
\pdv{u_{-2}}{x_2} &= 0, \\
\pdv{u_{-1}}{x_2} &= 1, \\
\pdv{u_0}{x_2} &= 0, \\
\pdv{u_1}{x_2} &= \pdv{u_1}{u_{-1}} \pdv{u_{-1}}{x_2} + \pdv{u_1}{u_0} \pdv{u_0}{x_2} = 1, \\
\pdv{u_2}{x_2} &= \pdv{u_2}{u_{-2}} \pdv{u_{-2}}{x_2} = 0, \\
\pdv{u_3}{x_2} &= \pdv{u_3}{u_2} \pdv{u_2}{x_2} = 0, \\
\pdv{u_4}{x_2} &= \pdv{u_4}{u_{-2}} \pdv{u_{-2}}{x_2} + \pdv{u_4}{u_1} \pdv{u_1}{x_2} = u_{-2} = x_1, \\
\pdv{u_5}{x_2} &= \pdv{u_5}{u_0} \pdv{u_0}{x_2} + \pdv{u_5}{u_3} \pdv{u_3}{x_2} = 0.
\end{align*}
$$

After a second iteration, we also know that $$\pdv{y_1}{x_2} = x_1$$ and $$\pdv{y_2}{x_2} = 0$$.

__Iteraton 3:__

$$
\begin{align*}
\pdv{u_{-2}}{x_3} &= 0, \\
\pdv{u_{-1}}{x_3} &= 0, \\
\pdv{u_0}{x_3} &= 1, \\
\pdv{u_1}{x_3} &= \pdv{u_1}{u_{-1}} \pdv{u_{-1}}{x_3} + \pdv{u_1}{u_0} \pdv{u_0}{x_3} = -1, \\
\pdv{u_2}{x_3} &= \pdv{u_2}{u_{-2}} \pdv{u_{-2}}{x_3} = 0, \\
\pdv{u_3}{x_3} &= \pdv{u_3}{u_2} \pdv{u_2}{x_3} = 0, \\
\pdv{u_4}{x_3} &= \pdv{u_4}{u_{-2}} \pdv{u_{-2}}{x_3} + \pdv{u_4}{u_1} \pdv{u_1}{x_3} = -u_{-2} = -x_1, \\
\pdv{u_5}{x_3} &= \pdv{u_5}{u_0} \pdv{u_0}{x_3} + \pdv{u_5}{u_3} \pdv{u_3}{x_3} = u_3 = \log(1 - x_1).
\end{align*}
$$

A third and final iteration yields the remaining $$\pdv{y_1}{x_3} = -x_1$$ and $$\pdv{y_2}{x_3} = \log(1 - x_1)$$.

Before drawing any conclusions, let us work through the same example again. This time around, we will perform the outside-in traversal of the chain rule.

### The Reverse Accumulation Mode

__Iteration 1:__

$$
\begin{align*}
\pdv{y_1}{u_5} &= 0, \\
\pdv{y_1}{u_4} &= 1, \\
\pdv{y_1}{u_3} &= \pdv{y_1}{u_5} \pdv{u_5}{u_3} = 0, \\
\pdv{y_1}{u_2} &= \pdv{y_1}{u_3} \pdv{u_3}{u_2} = 0, \\
\pdv{y_1}{u_1} &= \pdv{y_1}{u_4} \pdv{u_4}{u_1} = u_{-2} = x_1, \\
\pdv{y_1}{u_0} &= \pdv{y_1}{u_1} \pdv{u_1}{u_0} + \pdv{y_1}{u_5} \pdv{u_5}{u_0} = -u_{-2} = -x_1, \\
\pdv{y_1}{u_{-1}} &= \pdv{y_1}{u_1} \pdv{u_1}{u_{-1}} = u_{-2} = x_1, \\
\pdv{y_1}{u_{-2}} &= \pdv{y_1}{u_2} \pdv{u_2}{u_{-2}} + \pdv{y_1}{u_4} \pdv{u_4}{u_{-2}} = u_1 = x_2 - x_3.
\end{align*}
$$

Behold the power of backpropagation! Computing the partial derivative with respect to every intermediate variable once gives us $$\pdv{y_1}{x_1} = x_2 - x_3$$, $$\pdv{y_1}{x_2} = x_1$$, and $$\pdv{y_1}{x_3} = -x_1$$.

__Iteration 2:__

$$
\begin{align*}
\pdv{y_2}{u_5} &= 1, \\
\pdv{y_2}{u_4} &= 0, \\
\pdv{y_2}{u_3} &= \pdv{y_2}{u_5} \pdv{u_5}{u_3} = u_0 = x_3, \\
\pdv{y_2}{u_2} &= \pdv{y_2}{u_3} \pdv{u_3}{u_2} = u_0 \frac{1}{u_2} = x_3 \frac{1}{1 - x_1}, \\
\pdv{y_2}{u_1} &= \pdv{y_2}{u_4} \pdv{u_4}{u_1} = 0, \\
\pdv{y_2}{u_0} &= \pdv{y_2}{u_1} \pdv{u_1}{u_0} + \pdv{y_2}{u_5} \pdv{u_5}{u_0} = u_3 = \log(1 - x_1), \\
\pdv{y_2}{u_{-1}} &= \pdv{y_2}{u_1} \pdv{u_1}{u_{-1}} = 0, \\
\pdv{y_2}{u_{-2}} &= \pdv{y_2}{u_2} \pdv{u_2}{u_{-2}} + \pdv{y_2}{u_4} \pdv{u_4}{u_{-2}} = -u_0 \frac{1}{u_2} = -\frac{x_3}{1 - x_1}.
\end{align*}
$$

A second and final iteration concludes with $$\pdv{y_2}{x_1} = -x_3 / (1 - x_1)$$, $$\pdv{y_2}{x_2} = 0$$, and $$\pdv{y_2}{x_3} = \log(1 - x_1)$$. Do you start to recognize any patterns?

## Computational Complexity

Analyzing the pen-and-paper example, in the forward accumulation mode, we needed _three iterations_ because we had _three independent variables_. On the other hand, in the reverse accumulation mode, we only needed _two iterations_ because we had _two dependent variables_.

As a matter of fact, we can generalize the comparison of computational complexity to a generic function $$f \colon \R^n \to \R^m$$, where we would be able to draw the following conclusions:

1. In the forward accumulation mode, we would need $$n$$ iterations to compute the partial derivatives of the $$m$$ dependent variables with respect to the $$n$$ independent variables.
2. In the reverse accumulation mode, we would need $$m$$ iterations to compute the partial derivatives of the $$m$$ dependent variables with respect to the $$n$$ independent variables.

In closing, deep learning models may very well have trainable parameters in the millions but always only one cost function; hence, we always work with problems where $$n \gg m = 1$$, which is where backpropagation excels. Now, do you understand how backpropagation is able to reduce the time spent on computing gradients? 🏎
