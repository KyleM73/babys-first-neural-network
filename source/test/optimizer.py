import numpy as np


class GD:
    """Gradient Descent"""
    def __init__(self, mlp: object, eta: float) -> None:
        self.mlp = mlp
        self.eta = eta

    def step(self, dLdw: list, dLdw0: list) -> object:
        for layer, dLdw_i, dLdw0_i in zip(self.mlp.layers, dLdw, dLdw0):
            if hasattr(layer, "w"):
                axes = tuple([i for i in range(len(dLdw_i.shape) - 2)])
                layer.w -= self.eta * np.mean(dLdw_i, axis=axes).T
            if hasattr(layer, "w0"):
                axes = tuple([i for i in range(len(dLdw_i.shape) - 1)])
                layer.w0 -= self.eta * np.mean(dLdw0_i, axis=axes).T
        return self.mlp

    def __repr__(self) -> str:
        return "GradientDescent()"
