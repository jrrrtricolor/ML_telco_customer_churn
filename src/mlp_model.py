import logging

import torch
import torch.nn as nn


class MLPModel(nn.Module):
    """
    Rede Neural MLP simples para classificação de churn.

    Estrutura da rede:
    Entrada → Camada Oculta → ReLU → Camada de Saída

    Observação:
    - A saída retorna logits (valores brutos).
    - A função sigmoid NÃO é aplicada aqui, pois será tratada pela loss
      BCEWithLogitsLoss durante o treino.
    """

    def __init__(self, input_size: int, hidden_size: int = 32):
        """
        Inicializa a arquitetura da rede.

        Args:
            input_size (int): quantidade de variáveis de entrada (features)
            hidden_size (int): número de neurônios na camada oculta
        """
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.logger.info(
            "Inicializando MLPModel | input_size=%s | hidden_size=%s",
            input_size,
            hidden_size,
        )

        # Camada que conecta entrada → camada oculta
        self.hidden_layer = nn.Linear(input_size, hidden_size)

        # Função de ativação (introduz não-linearidade no modelo)
        self.activation = nn.ReLU()

        # Camada que conecta camada oculta → saída
        # Saída tem 1 neurônio (problema binário: churn ou não)
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Define como os dados passam pela rede.

        Etapas:
        1. Entrada passa pela camada oculta
        2. Aplicamos função de ativação (ReLU)
        3. Passa pela camada de saída
        4. Retorna logits (sem sigmoid)

        Args:
            x (Tensor): dados de entrada (features)

        Returns:
            Tensor: saída da rede (logits)
        """

        # Passo 1: transformação linear (entrada → hidden)
        x = self.hidden_layer(x)

        # Passo 2: aplica função de ativação
        x = self.activation(x)

        # Passo 3: transformação final (hidden → output)
        x = self.output_layer(x)

        # IMPORTANTE:
        # Não usamos sigmoid aqui, pois BCEWithLogitsLoss já faz isso internamente
        return x