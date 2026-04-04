# Shallow Neural Network

A neural network extends logistic regression by tacking multiple sigmoid units.

- Each node corresponds to two steps:
  1. Compute $z$ (linear step)  
  2. Compute $a = \sigma(z)$ (activation step)  

Instead of doing this once (as in logistic regression), a neural network repeats it multiple times across layers.

## Neural Network For Single Training Example

### Notation
- $a^{[0]} = x$ (one training example with a single input feature)  
- $z^{[1]}, a^{[1]}$ → layer 1 i.e. hidden layer activations
- $z^{[2]}, a^{[2]}$ → layer 2 i.e. prediction (output layer)
- $x^{(i)}$ → $i^{th}$ training example (in case of multiple training examples)

⚠️ Be careful: [ ] is for layers, ( ) is for training examples.

### Structure of a Neural Network
#### Layers
1. **Input Layer**  
   - $x$ contains one training example with 3 features.
   - $x$ has shape $\mathbb{R}^{1 \times 3}$
   - Features: $x_1, x_2, x_3$  
   - Denoted as activations $a^{[0]} = x$

2. **Hidden Layer**  
   - Contains 4 hidden units (not observed in the training data).  
   - Activations: $a^{[1]}$  
   - Example: with 4 hidden units,  
     $$
     a^{[1]} =
     \begin{bmatrix}
     a^{[1]}_1 \\
     a^{[1]}_2 \\
     a^{[1]}_3 \\
     a^{[1]}_4
     \end{bmatrix}
     \in \mathbb{R}^{4 \times 1}
     $$
   - In supervised learning:
      - Training data provides: $(X, y)$
      - Hidden layer values are not observed in the dataset.  
   - Hence, they are hidden.

3. **Output Layer**  
   - Single output unit.  
   - Activation: $a^{[2]} = \hat{y} \in \mathbb{R}$

### Parameters
Each layer (except input) has weights and biases:

- Hidden Layer (Layer 1):  
  $$
  W^{[1]} \in \mathbb{R}^{4 \times 3}, \quad b^{[1]} \in \mathbb{R}^{4 \times 1}
  $$

- Output Layer (Layer 2):  
  $$
  W^{[2]} \in \mathbb{R}^{1 \times 4}, \quad b^{[2]} \in \mathbb{R}^{1 \times 1}
  $$

Easy way to remember:

- Shape of $W$ is number of units in a layer $\times$ number of features of previous layer
- Shape of $b$ is number of units in a layer

### Forward propagation example
- **Layer 1**
   
   $z^{[1]} = W^{[1]}x + b^{[1]}$

   $a^{[1]} = \sigma(z^{[1]})$

- **Layer 2**

   $z^{[2]} = W^{[2]}a^{[1]} + b^{[2]}$

   $a^{[2]} = \sigma(z^{[2]})$

Final output:
$\hat{y} = a^{[2]}$

### Backward propagation (Intuition)
Just like logistic regression required a backward calculation for derivatives, neural networks also do this.

Backward steps (right-to-left):

Layer 2:

$da^{[2]} \rightarrow dz^{[2]} \rightarrow dW^{[2]}, db^{[2]}$

$dz^{[2]} = a^{[2]} - y$

$dW^{[2]} = dz^{[2]} a^{[1]T}$

$db^{[2]} = dz^{[2]}$

If you observe closely then you will see this step is quite similar to back propagation for logistic regression. Only the different is, while updating $dW^{[2]}$ we are not referring $X$ directly but we are referring to $a^{[1]}$ because we are working with multiple layers of neural network.

Layer 1:

$da^{[1]} \rightarrow dz^{[1]} \rightarrow dW^{[1]}, db^{[1]}$

$dz^{[1]} = W^{[2]T}dz^{[2]} * g^{[1]'}(z^{[1]})$

$dW^{[1]} = dz^{[1]} X^{T}$

$db^{[1]} = dz^{[1]}$

## Neural Network For Multiple Training Example
### Input layer

$X$ contains five training example with 3 features.

$X$ has shape $\mathbb{R}^{3 \times 5}$

We stack those 5 training examples horizontally in column vectors. 

So shape of $X$ is ${n_0 \times m}$. 

Where $n_0$ is features of layer 0 i.e. input layer and $m$ is total number of training examples.

### Hidden layer

Hidden layer i.e. layer $l = 1$ has 4 hidden units $n_l = n_1 = 4$

Shape of $W^{(1)}$ is ${4 \times 3}$ and shape of $b^{(1)}$ is ${4 \times 1}$, $W^{(1)}$ and $b^{(1)}$ are the parameters of layer $l = 1$

Shape of $W$ is $(n_l \times n_l\__1)$ where $l$ is the layer number and $n$ is number of units in a layer.

Shape of $Z^{(1)}$ is ${4 \times 5}$ (as result of dot product $Z^{(1)} = (W^{(1)})^T X + b$). 

Intuition can be as 5 training examples, each example gets processed through every unit.
So if we scan $Z^{(1)}$ vertically then we scan training examples and if we scan horizontally then we can output of each unit for a particular feature of each training example.

**General formula to remember**

If $X$ has shape ${n_0 \times m}$

And layer $l = 1$ has 4 hidden units $n_l = n_1 = 4$

Then $W^{(1)}$ takes shape of $n_1 \times n_0$

And $Z^{(1)}$ takes shape of $n_1 \times m$

And hence $A^{(1)}$ also takes shape of $n_1 \times m$ in this case ${4 \times 5}$

### Output layer

Output layer i.e. layer $l = 2$ has 1 unit $n_l = n_2 = 4$

Shape of $W^{(2)}$ is ${1 \times 4}$ and shape of $b^{(2)}$ is ${1 \times 1}$

Shape of $Z^{(2)}$ is ${1 \times 5}$ (as result of dot product $Z^{(2)} = (W^{(2)})^T A^{(1)} + b$)

Shape of $A^{(2)}$ is ${1 \times 5}$

So, in this way at the output layer we have got predictions for all the training example.

### Backward propagation

**Layer 2:**

$dA^{[2]} \rightarrow dZ^{[2]} \rightarrow dW^{[2]}, db^{[2]}$

$dZ^{[2]} = A^{[2]} - Y$

$dW^{[2]} = {\frac {1}{m}} * dZ^{[2]} A^{[1]T}$

$db^{[2]} = {\frac {1}{m}} * np.sum(dz^{[2]}, axis=1, keepdims = True)$

**Layer 1:**

$da^{[1]} \rightarrow dz^{[1]} \rightarrow dW^{[1]}, db^{[1]}$

$dZ^{[1]} = W^{[2]T}dZ^{[2]} * g^{[1]'}(Z^{[1]})$

$dW^{[1]} = {\frac {1}{m}} * dZ^{[1]} X^{T}$

$db^{[1]} = {\frac {1}{m}} * np.sum(dz^{[1]}, axis=1, keepdims = True)$

**Important note about dimensions of derivative terms in backward propagation**

- Dimension of $Z^{(2)}$ and $dZ^{(2)}$ is $n_2 \times m$
- Dimension of $W^{(2)}$ and $dW^{(2)}$ is $n_2 \times n_1$
- Dimension of $b^{(2)}$ and $db^{(2)}$ is $n_2 \times 1$
- Dimension of $Z^{(1)}$ and $dZ^{(1)}$ is $n_1 \times m$
- Dimension of $W^{(1)}$ and $dW^{(1)}$ is $n_1 \times n_2$
- Dimension of $b^{(1)}$ and $db^{(1)}$ is $n_1 \times 1$

**Note:** For any foo, dfoo has the same dimension as that of foo.

**This type of implementation can be done in Python by leveraging vectorized numpy operations.**

## Activation Functions in Neural Networks

When building a neural network, one of the important design choices is which activation function to use in the hidden layers and the output layer.  

So far, you may have only seen the sigmoid function, but other options often work much better. Let’s look at the most common ones.

Remember, when you implement **backpropagation** in a neural network, you need to compute the **derivative (slope)** of the activation functions. Let’s look at the common activation functions and their derivatives.

### Sigmoid

The sigmoid (or logistic) function maps inputs between 0 and 1:

$\sigma(z) = \frac{1}{1 + e^{-z}}$

- Useful when output needs to represent probabilities.  
- Historically popular, but has drawbacks:
  - Gradients vanish for very large or very small $z$.
  - Mean activation is not centered (tends toward 0.5).  

**Use case:** Output layer for binary classification.

#### Derivative of Sigmoid
The sigmoid function is:

$g(z) = \sigma(z) = \frac{1}{1 + e^{-z}}$

Its derivative can be shown (via calculus) to be:

$g'(z) = g(z)(1 - g(z))$

Sanity checks:
- If $z \gg 0$, then $g(z) \approx 1$, so $g'(z) \approx 0$  
- If $z \ll 0$, then $g(z) \approx 0$, so $g'(z) \approx 0$  
- If $z = 0$, then $g(z) = 0.5$, so $g'(z) = 0.25$  

In terms of the activation $a = g(z)$:

$g'(z) = a(1 - a)$

This is often used in implementations since \(a\) is already computed during forward propagation.

### Hyperbolic tangent (tanh)

The tanh function maps inputs between -1 and +1:

$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$

- Similar shape to sigmoid, but centered at zero.  
- Hidden units often train better since outputs are roughly zero-mean.  
- Generally performs strictly better than sigmoid for hidden layers.  

**Use case:** Hidden layers (sometimes).

#### Derivative of tanh
The tanh function is:

$g(z) = \tanh(z)$

Its derivative is:

$g'(z) = 1 - \tanh^2(z)$

Or, if $a = g(z)$:

$g'(z) = 1 - a^2$

Sanity checks:
- If \(z \gg 0\), \(\tanh(z) \approx 1\), so \(g'(z) \approx 0\).  
- If \(z \ll 0\), \(\tanh(z) \approx -1\), so \(g'(z) \approx 0\).  
- If \(z = 0\), \(\tanh(z) = 0\), so \(g'(z) = 1\).  

### Rectified linear unit (ReLU)

The ReLU is now the default activation function in deep learning:

$\text{ReLU}(z) = \max(0, z)$

- Derivative:
  - $1$ for $z > 0$  
  - $0$ for $z < 0$  
- Non-differentiable at $z = 0$, but this rarely matters in practice.  
- Pros:
  - Efficient to compute.
  - Avoids vanishing gradient problem in positive region.  
- Cons:
  - Gradient is zero for $z < 0$ (“dying ReLU” problem).  

**Use case:** Default choice for hidden layers.

#### Derivative of ReLU
The ReLU function is:

$g(z) = \text{ReLU}(z) = \max(0, z)$

Its derivative is:

$
g'(z) =
\begin{cases}
0, & z < 0 \\
1, & z > 0
\end{cases}
$

At \(z = 0\), the derivative is technically undefined, but in practice implementations set it to either 0 or 1. This works fine for gradient descent.

### Leaky ReLU

A small tweak to ReLU:

$\text{Leaky ReLU}(z) = \max(\alpha z, z), \quad \alpha \approx 0.01$

- Allows a small negative slope instead of zero.  
- Helps mitigate the “dying ReLU” issue.  
- Less common in practice, but can work better.  

#### Derivative of leaky ReLU
The Leaky ReLU function is:

$g(z) = \max(\alpha z, z), \quad \alpha \approx 0.01$

Its derivative is:

$
g'(z) =
\begin{cases}
\alpha, & z < 0 \\
1, & z > 0
\end{cases}
$

Again, the derivative is undefined at \(z = 0\), but setting it to either $\alpha$ or 1 in code is fine.

### General rules

- **Binary classification (output layer):** Use **sigmoid**.  
- **Hidden layers (default):** Use **ReLU**.  
- **Alternative hidden layers:** Try **tanh** or **Leaky ReLU** if ReLU doesn’t perform well.  

### Practical notes

- Sigmoid and tanh can both suffer from the **vanishing gradient problem** at large \(|z|\).  
- ReLU avoids this issue for positive \(z\), which often makes learning much faster.  
- There’s no universal “best” activation function. If in doubt:
  1. Start with ReLU for hidden layers.  
  2. Use sigmoid at the output for binary classification.  
  3. Experiment with alternatives (tanh, Leaky ReLU) and evaluate on a validation set.  

## Why does a neural network need a non-linear activation function?

For a neural network to compute interesting functions, it must use non-linear activation functions. Let’s see why.

### Linear (identity) activation function

Suppose we remove the non-linearity and just set:

$a^{[1]} = z^{[1]} \quad \text{or equivalently} \quad g(z) = z$

This is sometimes called the linear activation function, or more precisely, the identity activation function since it simply outputs its input.

If we continue with this idea:

$a^{[2]} = z^{[2]} = W^{[2]} a^{[1]} + b^{[2]}$

and since

$a^{[1]} = z^{[1]} = W^{[1]} x + b^{[1]}$

substituting gives:

$a^{[2]} = W^{[2]}(W^{[1]}x + b^{[1]}) + b^{[2]}$

$= (W^{[2]}W^{[1]})x + (W^{[2]}b^{[1]} + b^{[2]})$

$= W'x + b'$

So the output is still just a **linear function of the input**.

### Consequences

- With only linear activations, even if you stack many layers, the entire network is just computing a single linear function.  
- That means **deep linear networks are no more powerful than a single-layer linear model**.  
- Example: A linear hidden layer with a sigmoid output layer is no more expressive than standard logistic regression.
- Linear hidden layers are essentially useless. To model complex functions, we need **non-linear activations** (e.g., ReLU, tanh, sigmoid).

### When Is Linear Useful?

There is one common case: **regression tasks**.  

If \(y\) is a real-valued number (e.g., predicting house prices), then a linear activation at the **output layer** makes sense:

$g(z) = z$

- Outputs can range from $-\infty$ to $+\infty$, matching regression targets.  
- Hidden layers, however, should still use non-linear activations such as ReLU, tanh, or Leaky ReLU.  

Even then, sometimes ReLU is used at the output if predictions must be **non-negative** (e.g., housing prices).


