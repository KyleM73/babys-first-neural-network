import numpy as np

from loss import MSELoss
from mlp import MLP
from optimizer import GD


def f(x: np.ndarray) -> np.ndarray:
    return 2 * (x ** 3 - 1)


in_dim = 3
N = 1_000
epochs = 500
lr = 0.01
h_dims = [10, 10]

x = np.random.rand(N, in_dim)
y = f(x)
out_dim = y.shape[-1]
x_test = np.random.rand(N // 10, in_dim)
y_test = f(x_test)


mlp = MLP(in_dim=in_dim, out_dim=out_dim, hidden_dims=h_dims)
loss_fn = MSELoss()
optim_fn = GD(mlp, eta=lr)

print(f"\nNetwork: {mlp}")
print(f"Loss Function: {loss_fn}")
print(f"Optimizer: {optim_fn}\n")

for i in range(epochs):
    y_hat, latents = mlp.forward(x)
    loss = loss_fn.forward(y, y_hat)
    dLdy = loss_fn.backward(y, y_hat)
    dLdw, dLdw0 = mlp.backward(x, y_hat, dLdy, latents=latents)

    y_test_hat, _ = mlp.forward(x_test)
    test_loss = loss_fn.forward(y_test, y_test_hat)

    mlp = optim_fn.step(dLdw, dLdw0)
    print(f"[{i+1}] Loss: {loss:.4f}, Test Loss: {test_loss:.4f}")
