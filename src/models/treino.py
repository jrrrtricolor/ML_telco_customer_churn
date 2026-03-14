import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class Treino:

    def __init__(
        self,
        variaveis_explicaveis: pd.DataFrame,
        variavel_target: pd.Series,
        random_state: int = 42
    ):

        self.X = variaveis_explicaveis
        self.y = variavel_target
        self.random_state = random_state

        self.x_treino = None
        self.x_teste = None
        self.y_treino = None
        self.y_teste = None

    # --------------------------------------------------
    # Split dos dados
    # --------------------------------------------------

    def split_dados(self, test_size: float = 0.2) -> None:

        self.x_treino, self.x_teste, self.y_treino, self.y_teste = train_test_split(
            self.X,
            self.y,
            test_size=test_size,
            stratify=self.y,
            random_state=self.random_state
        )

        print("Dados divididos em treino e teste")

    # --------------------------------------------------
    # Criar modelos
    # --------------------------------------------------

    def criar_modelos(self) -> dict:

        modelos = {

            "dummy": DummyClassifier(strategy="most_frequent"),

            "decision_tree": DecisionTreeClassifier(
                random_state=self.random_state
            ),

            "random_forest": RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state
            ),

            "knn": KNeighborsClassifier(
                n_neighbors=5
            )
        }

        modelos_treinados = {}

        for nome, modelo in modelos.items():

            modelo.fit(self.x_treino, self.y_treino)

            modelos_treinados[nome] = modelo

            print(f"Modelo treinado: {nome}")

        return modelos_treinados

    # --------------------------------------------------
    # Avaliar modelos
    # --------------------------------------------------

    def avaliar_modelos(self, modelos: dict) -> pd.DataFrame:

        resultados = []

        for nome, modelo in modelos.items():

            y_pred = modelo.predict(self.x_teste)

            score = accuracy_score(self.y_teste, y_pred)

            resultados.append(
                {
                    "modelo": nome,
                    "accuracy": score
                }
            )

        df_resultados = pd.DataFrame(resultados)

        df_resultados = df_resultados.sort_values(
            by="accuracy",
            ascending=False
        )

        return df_resultados

    # --------------------------------------------------
    # Plotar árvore de decisão
    # --------------------------------------------------

    def plotar_arvore_decisao(self, modelo) -> None:

        plt.figure(figsize=(20, 10))

        plot_tree(
            modelo,
            filled=True,
            feature_names=self.X.columns,
            class_names=True
        )

        plt.title("Árvore de Decisão")

        plt.show()

    # --------------------------------------------------
    # Salvar modelo
    # --------------------------------------------------

    @staticmethod
    def salvar_modelo(modelo, nome_modelo: str, path: str = "../models/trained_models") -> None:

        os.makedirs(path, exist_ok=True)

        file_path = os.path.join(path, f"{nome_modelo}.joblib")

        joblib.dump(modelo, file_path)

        print(f"Modelo salvo em: {file_path}")