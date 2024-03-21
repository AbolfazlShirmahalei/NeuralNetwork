from typing import Optional

import numpy as np


class Layer:
    def __init__(
        self,
        input_size: int,
        output_size: int,
        weight: Optional[np.ndarray] = None,
        bias: Optional[np.ndarray] = None,
    ):
        self.input_size = input_size
        self.output_size = output_size

        if weight is None:
            self._weight = np.random.rand(input_size, output_size)
        else:
            assert weight.shape == (input_size, output_size), (
                f"Weight must be in the form of {(input_size, output_size)}"
                f", but it is in the form of {weight.shape}."
            )
            self._weight = weight

        if bias is None:
            self._bias = np.random.rand(output_size)
        else:
            assert bias.shape == (output_size,), (
                f"Weight must be in the form of {(output_size,)}"
                f", but it is in the form of {bias.shape}."
            )
            self._bias = bias

    @property
    def weight(self) -> np.ndarray:
        return self._weight

    @weight.setter
    def weight(self, weight: np.ndarray):
        assert weight.shape == (self.input_size, self.output_size), (
            f"New weight must be in the form of {(self.input_size, self.output_size)}"
            f", but it is in the form of {weight.shape}."
        )
        self._weight = weight

    @property
    def bias(self) -> np.ndarray:
        return self._bias

    @bias.setter
    def bias(self, bias: np.ndarray):
        assert bias.shape == (self.output_size,), (
            f"New bias must be in the form of {(self.output_size,)}"
            f", but it is in the form of {bias.shape}."
        )
        self._bias = bias
