from tinygrad.tensor import Tensor
import numpy as np

# Create tensors with requires_grad set to True to enable gradient tracking
x = Tensor(np.arange(2, 20).reshape((3, 6)), requires_grad=True)  # Shape (3, 6)
y = Tensor(np.arange(2, 20).reshape((6, 3)), requires_grad=True)  # Shape (6, 3)



# Perform matrix multiplication
prod = x.matmul(y)  # Resulting shape will be (3, 3)

# Define a "loss" as the sum of the elements in the product tensor
prodsum = prod.sum()

# Print the results before backpropagation
print("Product Matrix (prod):")
print(prod.numpy())
print("\nSum of Product Matrix (prodsum):", prodsum.numpy())

# Perform backpropagation to compute gradients
prodsum.backward()

# Print the gradients of x and y
print("\nGradients of x:")
print(x.grad.numpy())
print("\nGradients of y:")
print(y.grad.numpy())

# Simulate a simple optimization step (gradient descent)
learning_rate = 2
x = x - x.grad * learning_rate
y = y - y.grad * learning_rate

# Recompute the product after the update
new_prod = x.matmul(y)

# Recompute the sum of the resulting tensor
new_prodsum = new_prod.sum()

# Print the results after the optimization step
print("\nUpdated Product Matrix (new_prod):")
print(new_prod.numpy())
print("\nUpdated Sum of Product Matrix (new_prodsum):", new_prodsum.numpy())