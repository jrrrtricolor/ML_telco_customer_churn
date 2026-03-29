import logging
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim


class Trainer:
    """
    Classe responsável por treinar modelos de Machine Learning.

    Suporta dois tipos de modelos:
    - sklearn (fit/predict)
    - PyTorch (MLP com loop de treino manual)
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def treinar_modelos(
        self,
        modelos: Dict[str, object],
        X_train,
        y_train,
    ) -> Dict[str, object]:
        """
        Treina múltiplos modelos.

        Args:
            modelos: dicionário com nome e instância dos modelos
            X_train: dados de treino (features)
            y_train: rótulos de treino

        Returns:
            dicionário com modelos treinados
        """

        modelos_treinados = {}

        for nome, modelo in modelos.items():
            self.logger.info(f"Iniciando treino do modelo: {nome}")

            modelo_treinado = self._treinar_modelo(modelo, X_train, y_train)

            modelos_treinados[nome] = modelo_treinado

        return modelos_treinados

    def _treinar_modelo(self, modelo, X_train, y_train):
        """
        Treina um único modelo.

        Detecta automaticamente se o modelo é sklearn ou PyTorch.
        """

        try:
            # Verifica se é modelo PyTorch
            if isinstance(modelo, torch.nn.Module):
                modelo = self._treinar_modelo_torch(modelo, X_train, y_train)
            else:
                modelo.fit(X_train, y_train)

            self.logger.info("Modelo treinado com sucesso.")

        except Exception as e:
            self.logger.error(f"Erro ao treinar o modelo: {e}")
            raise e

        return modelo

    def _treinar_modelo_torch(self, modelo, X_train, y_train):
        """
        Treina modelos PyTorch (MLP).

        Fluxo:
        - Converte dados para tensor
        - Executa loop de treino
        - Atualiza pesos via backpropagation
        """

        self.logger.info("Treinando modelo PyTorch (MLP)")

        modelo.train()

        # Converte DataFrame → numpy → tensor
        X_train = torch.tensor(X_train.values, dtype=torch.float32)
        y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

        # Função de perda (binário)
        criterion = nn.BCEWithLogitsLoss()

        # Otimizador
        optimizer = optim.Adam(modelo.parameters(), lr=0.001)

        epochs = 20

        for epoch in range(epochs):
            optimizer.zero_grad()

            # Forward pass
            outputs = modelo(X_train)

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

        self.logger.info("Treinamento PyTorch finalizado")

        return modelo

    def predict(self, modelo, X_test):
        """
        Gera previsões (0 ou 1).

        Funciona para sklearn e PyTorch.
        """

        # PyTorch
        if isinstance(modelo, torch.nn.Module):
            modelo.eval()

            X_test = torch.tensor(X_test.values, dtype=torch.float32)

            with torch.no_grad():
                outputs = modelo(X_test)
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).int()

            return preds.numpy().flatten()

        # sklearn
        return modelo.predict(X_test)

    def predict_proba(self, modelo, X_test):
        """
        Retorna probabilidades (necessário para AUC).

        Funciona para sklearn e PyTorch.
        """

        # PyTorch
        if isinstance(modelo, torch.nn.Module):
            modelo.eval()

            X_test = torch.tensor(X_test.values, dtype=torch.float32)

            with torch.no_grad():
                outputs = modelo(X_test)
                probs = torch.sigmoid(outputs)

            return probs.numpy().flatten()

        # sklearn
        return modelo.predict_proba(X_test)[:, 1]