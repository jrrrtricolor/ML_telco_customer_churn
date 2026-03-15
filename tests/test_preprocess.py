from pathlib import Path

from src.utils.arquivo import Arquivo


def test_arquivo_preparado_sem_nulos() -> None:
	root_dir = Path(__file__).resolve().parents[1]
	arquivo_processado = root_dir / "data" / "processed" / "telco_churn_encoded.csv"

	dados = Arquivo.carregar_dados(str(arquivo_processado))

	assert (
		not dados.isnull().values.any()
	), "O arquivo preparado possui valores nulos e precisa de tratamento."

