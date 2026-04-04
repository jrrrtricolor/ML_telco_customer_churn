import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from sklearn.base import BaseEstimator
from src.mlp_model import MLPModel

class SkLearnMLPModel(BaseEstimator):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)

    def fit(self, X, y):
        input_size = X.shape[1]
        hidden_size = 32

        # Camada que conecta entrada → camada oculta
        self.model_module = MLPModel(input_size, hidden_size)

        # Função de ativação (introduz não-linearidade no modelo)
        self.activation = nn.ReLU()

        # Camada que conecta camada oculta → saída
        # Saída tem 1 neurônio (problema binário: churn ou não)
        self.output_layer = nn.Linear(hidden_size, 1)

        self.model_module.train()

        # Converte DataFrame → numpy → tensor
        X_train = torch.tensor(X, dtype=torch.float32)
        y_train = torch.tensor(y, dtype=torch.float32).view(-1, 1)

        # Função de perda (binário)
        criterion = nn.BCEWithLogitsLoss()

        # Otimizador
        optimizer = optim.Adam(self.model_module.parameters(), lr=0.001)

        epochs = 20

        for epoch in range(epochs):
            optimizer.zero_grad()

            # Forward pass
            outputs = self.model_module(X_train)

            # Calcula erro
            loss = criterion(outputs, y_train)

            # Backpropagation
            loss.backward()

            # Atualiza pesos
            optimizer.step()

            if (epoch + 1) % 5 == 0:
                self.logger.info(
                    f"Epoch {epoch + 1}/{epochs} - Loss: {loss.item():.4f}"
                )

        self.model_module.eval()
        self.logger.info("Treinamento PyTorch finalizado")

        self.is_fitted_ = True

        return self

    def transform(self, X):
        return X

    def predict(self, X):
        self.model_module.eval()

        X_test = torch.tensor(X, dtype=torch.float32)

        with torch.no_grad():
            outputs = self.model_module(X_test)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).int()

        return preds.numpy().flatten()

    def predict_proba(self, X):
        self.model_module.eval()

        X_test = torch.tensor(X, dtype=torch.float32)

        with torch.no_grad():
            outputs = self.model_module(X_test)
            probs = torch.sigmoid(outputs)

        value = probs.numpy().flatten()
        value_inv = 1 - value

        return np.stack((value_inv, value), axis=-1)
