#Bibliotecas internas.
import src.pipeline as pipeline


if __name__ == "__main__":
    # Criar uma instância da pipeline
    minha_pipeline = pipeline.Pipeline()

    # Executar a pipeline
    resultados = minha_pipeline.executar()

    # Exibir os resultados
    print(resultados)