import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src.mlp_model import MLPModel


class SkLearnMLPModel(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        hidden_size: int = 32,
        lr: float = 0.001,
        epochs: int = 60,
        batch_size: int = 64,
        val_size: float = 0.2,
        patience: int = 8,
        random_state: int = 2711,
    ):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.hidden_size = hidden_size
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.val_size = val_size
        self.patience = patience
        self.random_state = random_state

    def fit(self, x, y):
        x_array = np.asarray(x, dtype=np.float32)
        y_array = np.asarray(y, dtype=np.float32)

        x_train, x_val, y_train, y_val = train_test_split(
            x_array,
            y_array,
            test_size=self.val_size,
            random_state=self.random_state,
            stratify=y_array,
        )

        torch.manual_seed(self.random_state)

        input_size = x_array.shape[1]
        self.model_module = MLPModel(input_size, self.hidden_size)

        train_dataset = TensorDataset(
            torch.tensor(x_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32).view(-1, 1),
        )

        val_x = torch.tensor(x_val, dtype=torch.float32)
        val_y = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model_module.parameters(), lr=self.lr)

        melhor_val_loss = float("inf")
        melhor_estado = None
        paciencia_atual = 0

        for epoch in range(self.epochs):
            self.model_module.train()
            soma_loss_treino = 0.0

            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = self.model_module(x_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                soma_loss_treino += float(loss.item())

            train_loss = soma_loss_treino / max(len(train_loader), 1)

            self.model_module.eval()
            with torch.no_grad():
                val_outputs = self.model_module(val_x)
                val_loss = float(criterion(val_outputs, val_y).item())

            if val_loss < melhor_val_loss:
                melhor_val_loss = val_loss
                melhor_estado = {
                    k: v.clone().detach()
                    for k, v in self.model_module.state_dict().items()
                }
                paciencia_atual = 0
            else:
                paciencia_atual += 1

            if (epoch + 1) % 5 == 0:
                self.logger.info(
                    "Epoch %s/%s - train_loss: %.4f - val_loss: %.4f",
                    epoch + 1,
                    self.epochs,
                    train_loss,
                    val_loss,
                )

            if paciencia_atual >= self.patience:
                self.logger.info(
                    "Early stopping ativado na epoch %s.", epoch + 1
                )
                break

        if melhor_estado is not None:
            self.model_module.load_state_dict(melhor_estado)

        self.classes_ = np.array([0, 1])
        self.n_features_in_ = x_array.shape[1]
        self._estimator_type = "classifier"

        self.model_module.eval()
        self.logger.info("Treinamento PyTorch finalizado")

        self.is_fitted_ = True

        return self

    def transform(self, x):
        return x

    def predict(self, x):
        self.model_module.eval()

        x_test = torch.tensor(np.asarray(x), dtype=torch.float32)

        with torch.no_grad():
            outputs = self.model_module(x_test)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).int()

        return preds.numpy().flatten()

    def predict_proba(self, x):
        self.model_module.eval()

        x_test = torch.tensor(np.asarray(x), dtype=torch.float32)

        with torch.no_grad():
            outputs = self.model_module(x_test)
            probs = torch.sigmoid(outputs)

        value = probs.numpy().flatten()
        value_inv = 1 - value

        return np.stack((value_inv, value), axis=-1)
