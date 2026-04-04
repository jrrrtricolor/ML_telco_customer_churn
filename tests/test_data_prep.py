import pandas as pd

from src.data_prep import DataPreprocessor


def _criar_df_telco_exemplo() -> pd.DataFrame:
    """Gera uma base pequena para validar fluxo de preparação."""
    return pd.DataFrame(
        {
            "customerID": ["0001", "0002", "0003", "0004", "0005", "0006"],
            "gender": ["Male", "Female", "Female", "Male", "Female", "Male"],
            "InternetService": ["DSL", "Fiber optic", "No", "DSL", "No", "Fiber optic"],
            "MonthlyCharges": [29.85, 56.95, 0.0, 42.30, 70.20, 90.10],
            "TotalCharges": ["29.85", "1889.5", " ", "108.15", "1400.7", "900.1"],
            "tenure": [1, 34, 2, 3, 20, 10],
            "Churn": ["No", "No", "Yes", "No", "Yes", "Yes"],
        }
    )


def test_normalizar_dados_aplica_mapeamento_de_churn():
    dados = _criar_df_telco_exemplo()
    prep = DataPreprocessor(pd_dataframe=dados.copy(), seed=42, test_size=0.2)

    dados_norm = prep.normalizar_dados()

    assert "InternetService" in dados_norm.columns
    assert set(dados_norm["Churn"].unique()).issubset({0, 1})


def test_preparar_gera_split_treino_teste_sem_quebrar():
    dados = _criar_df_telco_exemplo()
    prep = DataPreprocessor(pd_dataframe=dados.copy(), seed=42, test_size=0.4)

    # Evita escrita em disco no teste.
    prep.salvar_arquivo = lambda _df: None

    prep.preparar()

    assert prep.X_train is not None
    assert prep.X_test is not None
    assert prep.y_train is not None
    assert prep.y_test is not None

    # customerID deve ser removido do conjunto final.
    assert "customerID" not in prep.X.columns
