import logging
from typing import Dict


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
            modelo = modelo.fit(X_train, y_train)

            self.logger.info("Modelo treinado com sucesso.")

        except Exception as e:
            self.logger.error(f"Erro ao treinar o modelo: {e}")
            raise e

        return modelo

    def predict(self, modelo, X_test):
        """
        Gera previsões (0 ou 1).

        Funciona para sklearn e PyTorch.
        """
        
        return modelo.predict(X_test)

    def predict_proba(self, modelo, X_test):
        """
        Retorna probabilidades (necessário para AUC).

        Funciona para sklearn e PyTorch.
        """

        probs = modelo.predict_proba(X_test)

        self.logger.info(f"Probabilidades geradas para o modelo: {probs}")

        return probs[:, 1]
