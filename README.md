# Deep Learning

This directory contains resources and assignments related to deep learning concepts and architectures.

## Contents

- [Shallow Neural Network](#shallow-neural-network)
- [Deep Neural Network](#deep-neural-network)

## Shallow Neural Network

A **shallow neural network** typically consists of an input layer, a single hidden layer, and an output layer. It extends logistic regression by stacking multiple sigmoid (or other activation) units in a single hidden layer. Key points:

- **Input Layer:** Receives the input features.
- **Hidden Layer:** Contains a small number of units (neurons) and applies non-linear activation functions (e.g., sigmoid, tanh, ReLU).
- **Output Layer:** Produces the final prediction (e.g., probability for binary classification).

### Example Structure
- Input: $x \in \mathbb{R}^{n}$
- Hidden layer: $a^{[1]} = g(W^{[1]}x + b^{[1]})$
- Output: $a^{[2]} = \sigma(W^{[2]}a^{[1]} + b^{[2]})$

### Key Concepts
- **Forward Propagation:** Computes activations layer by layer.
- **Backward Propagation:** Computes gradients for learning parameters.
- **Activation Functions:** Sigmoid, tanh, ReLU, Leaky ReLU, etc.
- **Non-linearity:** Essential for modeling complex functions.

For detailed explanations, see [`shallow-neural-network/README.md`](shallow-neural-network/README.md).

## Deep Neural Network

A **deep neural network (DNN)** is an extension of shallow networks with multiple hidden layers between the input and output. This depth allows the network to learn hierarchical representations and model more complex functions.

- **Input Layer:** Receives raw features.
- **Multiple Hidden Layers:** Each layer extracts increasingly abstract features using non-linear activations.
- **Output Layer:** Produces the final prediction (classification, regression, etc.).

### Why Deep Networks?
- Can approximate highly complex, non-linear functions.
- Enable learning of hierarchical feature representations.
- Power modern AI applications (image recognition, NLP, etc.).

### Typical Structure
- Input: $x$
- Hidden layers: $a^{[l]} = g(W^{[l]}a^{[l-1]} + b^{[l]})$ for $l = 1, 2, ..., L$
- Output: $a^{[L+1]}$

For detailed explanations, see [`deep-neural-network/README.md`](deep-neural-network/README.md).
