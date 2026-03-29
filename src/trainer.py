import logging


class Trainer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)


    def _treinar_modelo(self, modelo, X_train, y_train):
        try:
            modelo.fit(X_train, y_train)
            self.logger.info("Modelo treinado com sucesso.")
        except Exception as e:
            self.logger.error(f"Erro ao treinar o modelo: {e}")
            raise e

        return modelo

    def treinar_modelos(self, modelos: dict, X_train, y_train) -> dict:
        #Treina uma lista de modelos.

        modelos_treinados = {}

        for nome, modelo in modelos.items():
            self.logger.info(f"Iniciando treino: {nome}")

            modelo_treinado = self._treinar_modelo(modelo, X_train, y_train)

            modelos_treinados[nome] = modelo_treinado

        self.logger.info("Treinamento concluído")

        return modelos_treinados

