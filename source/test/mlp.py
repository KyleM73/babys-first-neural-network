import numpy as np


class ReLU:
    """Rectified Linear Unit"""
    def forward(
            self,
            x: np.ndarray
    ) -> np.ndarray:
        return np.clip(x, a_min=0, a_max=None)

    def backward(
            self,
            x: np.ndarray,
            y: np.ndarray,
            dLdy: np.ndarray,
    ) -> np.ndarray:
        return x >= 0

    def __repr__(self) -> str:
        return "ReLU()"

    @property
    def num_params(self) -> int:
        return 0


class Linear:
    """Linear Layer"""
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            bias: bool = True,
    ) -> None:
        self.m = in_dim
        self.n = out_dim
        self.bias = bias
        self.last = False
        # sample initial weights w ~ N(0, 1/m)
        self.w = np.random.randn(self.m, self.n) / self.m
        self.w0 = np.random.randn(self.n) / self.m
        if not self.bias:
            self.w0 *= 0

    def forward(
            self,
            x: np.ndarray,
    ) -> np.ndarray:
        """Computes the forward pass of a linear layer

        Args:
            x (np.ndarray): layer input [..., m]

        Returns:
            y (np.ndarray): layer output [..., n]
        """
        return np.einsum("nm,...m->...n", self.w.T, x) + self.w0

    def backward(
            self,
            x: np.ndarray,
            y: np.ndarray,
            dLdy: np.ndarray,
    ) -> np.ndarray:
        """Computes the derivative wrt the input

        Args:
            x (np.ndarray): layer input [..., m]
            y (np.ndarray): layer output [..., n]
            dLdy (np.ndarray): layer gradient wrt the output [..., n]

        Returns:
            dLdx (np.ndarray): layer gradient wrt the input [m, n]
        """
        dydx = self.w.T
        return dLdy @ dydx
        # return np.einsum("...n,nm->...m", dLdy, dydx)

    def backward_w(
            self,
            x: np.ndarray,
            dLdy: np.ndarray,
    ) -> np.ndarray:
        """Computes the gradient wrt the weight w

        Args:
            x (np.ndarray): layer input [..., m]
            dLdy (np.ndarray): layer gradient wrt the output [..., n]

        Returns:
            dLdw (np.ndarray): layer gradient wrt the weight [..., m, n]
        """
        dydw = x  # x.T = x for flattened singleton dim
        return np.einsum("...n,...m->...nm", dLdy, dydw)

    def backward_w0(
            self,
            x: np.ndarray,
            dLdy: np.ndarray,
    ) -> np.ndarray:
        """Computes the gradient wrt the bias w0

        Args:
            x (np.ndarray): layer input [..., m]
            dLdy (np.ndarray): layer gradient wrt the output [..., n]

        Returns:
            dLdw0 (np.ndarray): layer gradient wrt the bias [..., n]
        """
        return dLdy if self.bias else 0 * dLdy

    def __repr__(self, activation: str = None) -> str:
        return "Linear(in={m}, out={n}, bias={bias}, activation={act})".format(
            m=self.m, n=self.n, bias=self.bias, act=activation
        )

    @property
    def num_params(self) -> int:
        return self.w.size + self.w0.size


class MLP:
    """Multi Layer Perceptron"""
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            hidden_dims: list = [10, 10],
            activation: type = ReLU,
    ) -> None:
        self.dims = hidden_dims + [out_dim]
        self.activation = activation()
        self.layers = [Linear(in_dim, self.dims[0])]
        for i in range(1, len(self.dims)):
            m, n = self.dims[i-1], self.dims[i]
            self.layers.append(self.activation)
            self.layers.append(Linear(m, n))
        self.params = sum([layer.num_params for layer in self.layers])

    def forward(
            self,
            x: np.ndarray,
    ) -> np.ndarray:
        """Computes the forward pass of the MLP

        Args:
            x (np.ndarray): MLP input [..., m]

        Returns:
            y (np.ndarray): MLP output [..., n]
        """
        latents = []
        for layer in self.layers:
            latents.append(x)
            x = layer.forward(x)
        return x, latents

    def backward(
            self,
            x: np.ndarray,
            y: np.ndarray,
            dLdy: np.ndarray,
            latents: list,
    ) -> tuple[list, list]:
        """Computes the derivative wrt each layer

        Args:
            x (np.ndarray): layer input [..., m]
            y (np.ndarray): layer output [..., n]
            dLdy (np.ndarray): layer gradient wrt the output [..., n]
            latents (list): intermediate layer outputs [len(self.layers)]

        Returns:
            dLdw (list): layer gradients wrt the weights
            dLdw0 (list): layer gradients wrt the biases
        """
        dLdw = []
        dLdw0 = []
        for layer, x in zip(reversed(self.layers), reversed(latents)):
            dLdx = layer.backward(x, y, dLdy)
            if hasattr(layer, "w"):
                dLdw_i = layer.backward_w(x, dLdy)
                dLdw.append(dLdw_i)
            else:
                dLdw.append(None)
            if hasattr(layer, "w0"):
                dLdw0_i = layer.backward_w0(x, dLdy)
                dLdw0.append(dLdw0_i)
            else:
                dLdw0.append(None)
            dLdy = dLdx
        return reversed(dLdw), reversed(dLdw0)

    def __repr__(self) -> str:
        msg = "MLP(\n"
        act = self.activation
        i = 0
        for layer in self.layers:
            if isinstance(layer, Linear):
                if layer == self.layers[-1]:
                    act = None
                msg += f"   [{i}] " + layer.__repr__(act) + "\n"
                i += 1
        msg += f"), Total Parameters: {self.params}"
        return msg

    @property
    def num_params(self) -> int:
        return self.params
