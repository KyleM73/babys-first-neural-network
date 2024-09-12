import numpy as np


class MSELoss:
    def forward(self, y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
        """Compute the mean squared error between the estimate and ground truth

        Args:
            y (np.ndarray): ground truth [..., n]
            y_hat (np.ndarray): estimate [..., n]

        Returns:
            loss np.ndarray: squared error loss, summed over batch [n]
        """
        return np.mean(np.square(y - y_hat))
    # np.einsum("...n->n", (y_hat - y)**2)

    def backward(self, y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
        """Compute the gradient of the loss wrt the estimate

        Args:
            y (np.ndarray): ground truth [..., n]
            y_hat (np.ndarray): estimate [..., n]

        Returns:
            dLdy_hat (np.ndarray): gradient wrt the estimate [..., n]
        """
        return -2 * (y - y_hat)

    def __repr__(self) -> str:
        return "MSELoss()"
