# Neural Networks from Scratch

Two complementary implementations exploring the fundamentals of neural networks and automatic differentiation, both applied to a binary classification task on the UCI Cleveland Heart Disease dataset.

## Projects

### MLP via Linear Algebra (`NeuralNet_LinearAlgebra.ipynb`)
A feedforward neural network built entirely through explicit matrix operations using NumPy — no autograd, no frameworks. Every forward pass, gradient calculation, and weight update is written out directly, making the linear algebra mechanics fully transparent.

**Includes:** mini-batch SGD, Xavier initialisation, k-fold cross validation, and a full evaluation suite (accuracy, F1, AUC, ROC curve, training curves).

### minigrad (`Minigrad.ipynb`)
Karpathy's scalar-origin autograd engine, 'micrograd', is extended to support NumPy arrays as the core value element. This allows batched gradient descent natively without explicit looping, improving efficiency significantly over a pure scalar implementation. A neural network built on top of minigrad is trained on the same heart disease dataset.

**Includes:**  reverse-mode autodiff, batched backpropagation, training curves.

## Motivation
The two projects are deliberately complementary. The MLP makes the underlying linear algebra explicit — you can see exactly what each operation does to the data. minigrad abstracts that away through automatic differentiation, showing how autograd engines work under the hood. Together they cover the same theoretical ground from two different angles.

## Requirements
```python
numpy
matplotlib
```
