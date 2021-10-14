---
title: Why do we use backpropagation?
---

To understand why it almost always is best to use backpropagation for training neural networks, we will compare the computational complexity between applying the chain rule in forward and reverse accumulation modes.

## Forward and reverse accumulation modes

Suppose we have a function

$$
y = f(g(h(x))).
$$

Let us decompose the function with the help of intermediate variables:

$$
\begin{aligned}
u_0 & = x, \\
u_1 & = h(u_0), \\
u_2 & = g(u_1), \\
u_3 & = f(u_2) = y.
\end{aligned}
$$

To compute the derivative $$\dv{y}{x}$$, we can traverse the chain rule

1. inside-out or
2. outside-in.

We will start with inside-out traversal or the so-called forward accumulation mode:

$$
\begin{aligned}
\dv{u_0}{x} & = 1, \\
\dv{u_1}{x} & = \dv{u_1}{u_0} \dv{u_0}{x} = \dv{h(u_0)}{u_0}, \\
\dv{u_2}{x} & = \dv{u_2}{u_1} \dv{u_1}{x} = \dv{g(u_1)}{u_1} \dv{h(u_0)}{u_0}, \\
\dv{u_3}{x} & = \dv{u_3}{u_2} \dv{u_2}{x} = \dv{f(u_2)}{u_2} \dv{g(u_1)}{u_1} \dv{h(u_0)}{u_0}.
\end{aligned}
$$

By contrast, outside-in traversal means that we apply the chain rule in reverse accumulation mode, which more commonly is referred to as backpropagation:

$$
\begin{aligned}
\dv{y}{u_3} & = 1, \\
\dv{y}{u_2} & = \dv{y}{u_3} \dv{u_3}{u_2} = \dv{f(u_2)}{u_2}, \\
\dv{y}{u_1} & = \dv{y}{u_2} \dv{u_2}{u_1} = \dv{f(u_2)}{u_2} \dv{g(u_1)}{u_1}, \\
\dv{y}{u_0} & = \dv{y}{u_1} \dv{u_1}{u_0} = \dv{f(u_2)}{u_2} \dv{g(u_1)}{u_1} \dv{h(u_0)}{u_0}.
\end{aligned}
$$

Both methods reach

$$
\dv{y}{x} = \dv{u_3}{x} = \dv{y}{u_0} = \dv{f(u_2)}{u_2} \dv{g(u_1)}{u_1} \dv{h(u_0)}{u_0}
$$

using the same number of computations; however, this is not always the case, as we soon will find out. Moreover, we see that inside-out traversal computes the recursive relation

$$
\dv{u_i}{x} = \dv{u_i}{u_{i - 1}} \dv{u_{i - 1}}{x}.
$$

On the other hand, outside-in traversal computes the recursive relation

$$
\dv{y}{u_i} = \dv{y}{u_{i + 1}} \dv{u_{i + 1}}{u_i}.
$$

Now, let us move on to a function $$f: \R^3 \to \R^2$$, where it will be easier to compare the computational complexity between applying the chain rule in forward and reverse accumulation modes.

## $$f: \R^3 \to \R^2$$

To make a good comparison, we need a function with multiple dependent and independent variables. The following function fulfills that requirement:

$$
\begin{aligned}
y_1 & = x_1 (x_2 - x_3), \\
y_2 & = x_3 \log(1 - x_1).
\end{aligned}
$$

Next, to make derivative computations as simple as possible, after decomposition, we make sure we are left with only basic arithmetic operations and elementary functions:

$$
\begin{aligned}
u_{-2} & = x_1, \\
u_{-1} & = x_2, \\
u_0 & = x_3, \\
u_1 & = u_{-1} - u_0, \\
u_2 & = 1 - u_{-2}, \\
u_3 & = \log(u_2), \\
u_4 & = u_{-2} u_1 = y_1, \\
u_5 & = u_0 u_3 = y_2.
\end{aligned}
$$

All in all, we have been strategic about how we have defined intermediate variables. Now, we are ready to compute $$\pdv{y_1}{x_1}$$, $$\pdv{y_1}{x_2}$$, $$\pdv{y_1}{x_3}$$, $$\pdv{y_2}{x_1}$$, $$\pdv{y_2}{x_2}$$ and $$\pdv{y_2}{x_3}$$. Once again, let us start with inside-out traversal of the chain rule, i.e., forward accumulation mode.

### Forward accumulation mode

__Iteration 1:__

$$
\begin{aligned}
\pdv{u_{-2}}{x_1} & = 1, \\
\pdv{u_{-1}}{x_1} & = 0, \\
\pdv{u_0}{x_1} & = 0, \\
\pdv{u_1}{x_1} & = \pdv{u_1}{u_{-1}} \pdv{u_{-1}}{x_1} + \pdv{u_1}{u_0} \pdv{u_0}{x_1} = 0, \\
\pdv{u_2}{x_1} & = \pdv{u_2}{u_{-2}} \pdv{u_{-2}}{x_1} = -1, \\
\pdv{u_3}{x_1} & = \pdv{u_3}{u_2} \pdv{u_2}{x_1} = -\frac{1}{u_2} = -\frac{1}{1 - x_1}, \\
\pdv{u_4}{x_1} & = \pdv{u_4}{u_{-2}} \pdv{u_{-2}}{x_1} + \pdv{u_4}{u_1} \pdv{u_1}{x_1} = u_1 = x_2 - x_3, \\
\pdv{u_5}{x_1} & = \pdv{u_5}{u_0} \pdv{u_0}{x_1} + \pdv{u_5}{u_3} \pdv{u_3}{x_1} = -u_0 \frac{1}{u_2} = -\frac{x_3}{1 - x_1}.
\end{aligned}
$$

Note that a single pass through all intermediate variables, in forward accumulation mode, gives us both $$\pdv{y_1}{x_1} = x_2 - x_3$$ and $$\pdv{y_2}{x_1} = -\frac{x_3}{1 - x_1}$$.

__Iteration 2:__

$$
\begin{aligned}
\pdv{u_{-2}}{x_2} & = 0, \\
\pdv{u_{-1}}{x_2} & = 1, \\
\pdv{u_0}{x_2} & = 0, \\
\pdv{u_1}{x_2} & = \pdv{u_1}{u_{-1}} \pdv{u_{-1}}{x_2} + \pdv{u_1}{u_0} \pdv{u_0}{x_2} = 1, \\
\pdv{u_2}{x_2} & = \pdv{u_2}{u_{-2}} \pdv{u_{-2}}{x_2} = 0, \\
\pdv{u_3}{x_2} & = \pdv{u_3}{u_2} \pdv{u_2}{x_2} = 0, \\
\pdv{u_4}{x_2} & = \pdv{u_4}{u_{-2}} \pdv{u_{-2}}{x_2} + \pdv{u_4}{u_1} \pdv{u_1}{x_2} = u_{-2} = x_1, \\
\pdv{u_5}{x_2} & = \pdv{u_5}{u_0} \pdv{u_0}{x_2} + \pdv{u_5}{u_3} \pdv{u_3}{x_2} = 0.
\end{aligned}
$$

After a second iteration, we also know that $$\pdv{y_1}{x_2} = x_1$$ and $$\pdv{y_2}{x_2} = 0$$.

__Iteraton 3:__

$$
\begin{aligned}
\pdv{u_{-2}}{x_3} & = 0, \\
\pdv{u_{-1}}{x_3} & = 0, \\
\pdv{u_0}{x_3} & = 1, \\
\pdv{u_1}{x_3} & = \pdv{u_1}{u_{-1}} \pdv{u_{-1}}{x_3} + \pdv{u_1}{u_0} \pdv{u_0}{x_3} = -1, \\
\pdv{u_2}{x_3} & = \pdv{u_2}{u_{-2}} \pdv{u_{-2}}{x_3} = 0, \\
\pdv{u_3}{x_3} & = \pdv{u_3}{u_2} \pdv{u_2}{x_3} = 0, \\
\pdv{u_4}{x_3} & = \pdv{u_4}{u_{-2}} \pdv{u_{-2}}{x_3} + \pdv{u_4}{u_1} \pdv{u_1}{x_3} = -u_{-2} = -x_1, \\
\pdv{u_5}{x_3} & = \pdv{u_5}{u_0} \pdv{u_0}{x_3} + \pdv{u_5}{u_3} \pdv{u_3}{x_3} = u_3 = \log(1 - x_1).
\end{aligned}
$$

A third and final iteration yields $$\pdv{y_1}{x_3} = -x_1$$ and $$\pdv{y_2}{x_3} = \log(1 - x_1)$$. Before drawing any conclusion, let us work through the same example again. This time, we will traverse the chain rule outside-in.

### Reverse accumulation mode

__Iteration 1:__

$$
\begin{aligned}
\pdv{y_1}{u_5} & = 0, \\
\pdv{y_1}{u_4} & = 1, \\
\pdv{y_1}{u_3} & = \pdv{y_1}{u_5} \pdv{u_5}{u_3} = 0, \\
\pdv{y_1}{u_2} & = \pdv{y_1}{u_3} \pdv{u_3}{u_2} = 0, \\
\pdv{y_1}{u_1} & = \pdv{y_1}{u_4} \pdv{u_4}{u_1} = u_{-2} = x_1, \\
\pdv{y_1}{u_0} & = \pdv{y_1}{u_1} \pdv{u_1}{u_0} + \pdv{y_1}{u_5} \pdv{u_5}{u_0} = -x_1, \\
\pdv{y_1}{u_{-1}} & = \pdv{y_1}{u_1} \pdv{u_1}{u_{-1}} = x_1, \\
\pdv{y_1}{u_{-2}} & = \pdv{y_1}{u_2} \pdv{u_2}{u_{-2}} + \pdv{y_1}{u_4} \pdv{u_4}{u_{-2}} = u_1 = x_2 - x_3.
\end{aligned}
$$

Behold the power of backpropagation! A single pass through all intermediate variables, in reverse accumulation mode, gives us $$\pdv{y_1}{x_1} = x_2 - x_3$$, $$\pdv{y_1}{x_2} = x_1$$ and $$\pdv{y_1}{x_3} = -x_1$$.

__Iteration 2:__

$$
\begin{aligned}
\pdv{y_2}{u_5} & = 1, \\
\pdv{y_2}{u_4} & = 0, \\
\pdv{y_2}{u_3} & = \pdv{y_2}{u_5} \pdv{u_5}{u_3} = u_0 = x_3, \\
\pdv{y_2}{u_2} & = \pdv{y_2}{u_3} \pdv{u_3}{u_2} = u_0 \frac{1}{u_2} = x_3 \frac{1}{1 - x_1}, \\
\pdv{y_2}{u_1} & = \pdv{y_2}{u_4} \pdv{u_4}{u_1} = 0, \\
\pdv{y_2}{u_0} & = \pdv{y_2}{u_1} \pdv{u_1}{u_0} + \pdv{y_2}{u_5} \pdv{u_5}{u_0} = u_3 = \log(1 - x_1), \\
\pdv{y_2}{u_{-1}} & = \pdv{y_2}{u_1} \pdv{u_1}{u_{-1}} = 0, \\
\pdv{y_2}{u_{-2}} & = \pdv{y_2}{u_2} \pdv{u_2}{u_{-2}} + \pdv{y_2}{u_4} \pdv{u_4}{u_{-2}} = -u_0 \frac{1}{u_2} = -\frac{x_3}{1 - x_1}.
\end{aligned}
$$

A second and final iteration concludes with $$\pdv{y_2}{x_1} = -\frac{x_3}{1 - x_1}$$, $$\pdv{y_2}{x_2} = 0$$ and $$\pdv{y_2}{x_3} = \log(1 - x_1)$$. Do you recognize any patterns?

## Computational complexity

Analyzing the pen-and-paper example, in forward accumulation mode, we needed _three iterations_ because we had _three independent variables_. On the other hand, in reverse accumulation mode, we only needed _two iterations_ because we had _two dependent variables_.

Next, let us generalize the comparison of computational complexity to a function $$f: \R^n \to \R^m$$, where we would recognize the following patterns:

1. In forward accumulation mode, we would need $$n$$ iterations to compute the derivatives of the $$m$$ dependent variables with respect to the $$n$$ independent variables.
2. In reverse accumulation mode, we would need $$m$$ iterations to compute the derivatives of the $$m$$ dependent variables with respect to the $$n$$ independent variables.

Subsequently, since deep learning models can have trainable parameters in the millions, but usually only one or a few cost functions, we almost always work with problems where $$n \gg m$$. Now, do you understand why backpropagation is for the win? 🐌⚡️
