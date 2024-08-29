# Tensor-Operations-and-Gradient-Descent-with-Tinygrad

This repository contains a Python script that demonstrates fundamental concepts in matrix operations, gradient computation, and optimization using the tinygrad library. The code showcases how to perform matrix multiplication, calculate gradients, and apply a simple gradient descent step to update tensors.

Overview
The code provided in this repository performs the following steps:

Create Tensors: Initializes two tensors x and y with gradient tracking enabled.
Matrix Multiplication: Computes the product of the two tensors.
Loss Calculation: Defines a loss function as the sum of the elements in the product matrix.
Gradient Computation: Calculates gradients of the tensors with respect to the loss.
Optimization: Updates the tensors using gradient descent and a specified learning rate.
Recalculation: Computes the product and sum of the updated tensors to observe the effects of the optimization step.


Requirements
  Python 3.x
  tinygrad library (installation instructions below)
  numpy library
