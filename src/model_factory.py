import logging

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from src.sklearn_mlp_model import SkLearnMLPModel
from src.sklearn_pipeline import criar_pipeline_modelo


class ModelFactory:
    def __init__(self, seed: int = 42):
        self.logger = logging.getLogger(__name__)
        self.seed = seed


    def criar_modelo_dummy(self):
        modelo = DummyClassifier(strategy="most_frequent")
        return criar_pipeline_modelo(modelo)


    def criar_modelo_decision_tree(self):
        modelo = DecisionTreeClassifier(random_state=self.seed)
        return criar_pipeline_modelo(modelo)


    def criar_modelo_random_forest(self):
        modelo = RandomForestClassifier(random_state=self.seed)
        return criar_pipeline_modelo(modelo)


    def criar_modelo_knn(self):
        modelo = KNeighborsClassifier()
        return criar_pipeline_modelo(modelo)


    def criar_modelo_logistic_regression(self):
        modelo = LogisticRegression(random_state=self.seed, max_iter=1000)
        return criar_pipeline_modelo(modelo)


    def criar_modelo_mlp(self):
        modelo = SkLearnMLPModel(random_state=self.seed)
        return criar_pipeline_modelo(modelo)


    def criar_modelos(self) -> dict:
        return {
            "dummy": self.criar_modelo_dummy(),
            "decision_tree": self.criar_modelo_decision_tree(),
            "random_forest": self.criar_modelo_random_forest(),
            "knn": self.criar_modelo_knn(),
            "logistic_regression": self.criar_modelo_logistic_regression(),
            "MLP": self.criar_modelo_mlp(),
        }
