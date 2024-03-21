from abc import ABC, abstractmethod
from typing import List, Tuple, Union, Generator

import numpy as np
from tqdm import tqdm


class NN(ABC):
    def __init__(
        self,
        hidden_layers: Union[List[int], Tuple[int]],
        batch_size: int,
    ):
        self._hidden_layers = hidden_layers
        self._batch_size = batch_size
        self._layers = None

        self._loss_history = {}
        self._epoch = -1

        self._X_size = None

    @abstractmethod
    def _check_y(self, y: np.ndarray):
        pass

    @abstractmethod
    def _prepare_y(self, y: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def _set_initial_weights(self):
        pass

    @abstractmethod
    def _calculate_loss(
        self,
        y: np.ndarray,
        predicted_y: np.ndarray,
    ) -> float:
        pass

    @abstractmethod
    def _update_weights(
        self,
        y: np.ndarray,
        predicted_outputs: List[np.ndarray],
    ):
        pass

    @abstractmethod
    def predict(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        pass

    def _predict_model_outputs(
        self,
        X: np.ndarray,
        save_hidden_layer: bool = True,
    ) -> List[np.ndarray]:
        pass

    @abstractmethod
    def _check_X(self, X: np.ndarray):
        assert len(X.shape) == 2, (
            f"X should be 2D, it is {len(X.shape)}D."
        )
        if self._X_size is None:
            self._X_size = int(X.shape[1])
        else:
            assert X.shape[1] == self._X_size, (
                f"X must be in the form of (n, {self._X_size})"
                f"but it is in the form of (n, {X.shape[1]})."
            )

    def _data_loader(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Generator:
        indexes = np.arange(len(X))
        permuted_indexes = np.random.permutation(indexes)
        i = 0
        while i < len(y):
            selected_indexes = permuted_indexes[i: i + self._batch_size]
            yield X[selected_indexes], y[selected_indexes]
            i += self._batch_size

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int,
    ):
        _X = np.asanyarray(X)
        _y = np.asanyarray(y)

        assert len(_X) == len(_y), (
            f"the size of X and y must be equal, but {len(X)}(#X) != {len(y)}(#y)."
        )
        self._check_X(X=_X)
        self._check_y(y=_y)

        _y = self._prepare_y(y=_y)

        if self._epoch == -1:
            self._set_initial_weights()

        for _ in range(epochs):
            loss = 0
            count = 0
            for batch_X, batch_y in tqdm(
                self._data_loader(X=_X, y=_y),
                desc=f"Epoch {self._epoch + 1}",
            ):
                batch_predicted_outputs = self._predict_model_outputs(X=batch_X)
                batch_predicted_y = batch_predicted_outputs[-1]
                batch_loss = self._calculate_loss(
                    y=batch_y,
                    predicted_y=batch_predicted_y,
                )

                loss += batch_loss
                count += len(batch_X)

                self._update_weights(
                    batch_y,
                    batch_predicted_outputs,
                )

            self._epoch += 1
            loss = loss / count
            self._loss_history[self._epoch] = loss
            print(f"Loss: {round(loss, 5)}")
