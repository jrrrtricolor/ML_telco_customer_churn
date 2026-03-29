import logging
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


class ModelFactory:
    def __init__(self, seed: int = 42):
        self.logger = logging.getLogger(__name__)
        self.seed = seed


    def criar_modelo_dummy(self):
        return DummyClassifier(strategy="most_frequent")


    def criar_modelo_decision_tree(self):
        return DecisionTreeClassifier(random_state=self.seed)


    def criar_modelo_random_forest(self):
        return RandomForestClassifier(random_state=self.seed)


    def criar_modelo_knn(self):
        return KNeighborsClassifier()


    def criar_modelos(self) -> dict:
        return {
            "dummy": DummyClassifier(strategy="most_frequent"),
            "decision_tree": DecisionTreeClassifier(random_state=self.seed),
            "random_forest": RandomForestClassifier(random_state=self.seed),
            "knn": KNeighborsClassifier()
        }