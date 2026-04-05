from sklearn.pipeline import Pipeline

from src.model_factory import ModelFactory


def test_model_factory_retorna_modelos_esperados():
    factory = ModelFactory(seed=2711)

    modelos = factory.criar_modelos()

    assert set(modelos.keys()) == {
        "dummy",
        "decision_tree",
        "random_forest",
        "knn",
        "MLP",
    }


def test_modelos_sao_pipelines_sklearn():
    factory = ModelFactory(seed=2711)

    modelos = factory.criar_modelos()

    for modelo in modelos.values():
        assert isinstance(modelo, Pipeline)
        assert "imputer" in modelo.named_steps
        assert "scaler" in modelo.named_steps
        assert "modelo" in modelo.named_steps
