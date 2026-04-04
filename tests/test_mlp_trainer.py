import numpy as np
import pandas as pd
import torch
from sklearn.tree import DecisionTreeClassifier

from src.mlp_model import MLPModel
from src.sklearn_mlp_model import SkLearnMLPModel
from src.trainer import Trainer


def test_mlp_forward_retorna_shape_esperado():
    modelo = MLPModel(input_size=3, hidden_size=8)
    x = torch.randn(5, 3)

    saida = modelo(x)

    assert saida.shape == (5, 1)


def test_sklearn_mlp_fit_predict_predict_proba():
    x_np = np.array(
        [
            [0.1, 1.0, 0.2],
            [0.2, 0.9, 0.1],
            [0.3, 0.8, 0.2],
            [0.9, 0.2, 0.8],
            [1.0, 0.1, 0.9],
            [1.1, 0.0, 1.0],
        ]
    )
    y_np = np.array([0, 0, 0, 1, 1, 1])

    modelo = SkLearnMLPModel().fit(x_np, y_np)

    y_pred = modelo.predict(x_np)
    y_prob = modelo.predict_proba(x_np)

    assert len(y_pred) == len(x_np)
    assert set(y_pred).issubset({0, 1})
    assert y_prob.shape == (len(x_np), 2)


def test_trainer_treina_modelo_sklearn_padrao():
    x_df = pd.DataFrame(
        {
            "f1": [0.1, 0.2, 0.3, 0.9, 1.0, 1.1, 0.4, 0.8],
            "f2": [1.0, 0.9, 0.8, 0.2, 0.1, 0.0, 0.7, 0.3],
            "f3": [0.2, 0.1, 0.2, 0.8, 0.9, 1.0, 0.3, 0.7],
        }
    )
    y = pd.Series([0, 0, 0, 1, 1, 1, 0, 1])

    trainer = Trainer()
    modelo = DecisionTreeClassifier(random_state=42)

    treinados = trainer.treinar_modelos({"arvore": modelo}, x_df, y)
    modelo_treinado = treinados["arvore"]

    y_pred = trainer.predict(modelo_treinado, x_df)

    assert len(y_pred) == len(x_df)
    assert set(y_pred).issubset({0, 1})
