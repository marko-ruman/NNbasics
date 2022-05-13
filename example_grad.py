import torch

x = torch.tensor([1.0], requires_grad=True)
y = x + 2
z = y * y * y * 2

z.backward()     # automatically calculates the gradient
print(x.grad)