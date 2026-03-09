"""
Testes básicos de sanidade para verificar que a estrutura do projeto está correta.
"""
import importlib
import os


def test_src_pacote_importavel():
    """Verifica que o pacote src pode ser importado."""
    modulo = importlib.import_module("src")
    assert modulo is not None


def test_diretorios_obrigatorios_existem():
    """Verifica que todos os diretórios obrigatórios do projeto existem."""
    base = os.path.dirname(os.path.dirname(__file__))
    diretorios = [
        "src",
        "src/data",
        "src/features",
        "src/models",
        "src/evaluation",
        "src/utils",
        "data/raw",
        "data/processed",
        "models/trained_models",
        "notebooks",
        "tests",
        "docs",
    ]
    for diretorio in diretorios:
        caminho = os.path.join(base, diretorio)
        assert os.path.isdir(caminho), f"Diretório obrigatório ausente: {diretorio}"


def test_arquivos_configuracao_existem():
    """Verifica que os arquivos de configuração do projeto existem."""
    base = os.path.dirname(os.path.dirname(__file__))
    arquivos = [
        "pyproject.toml",
        ".gitignore",
        "README.md",
    ]
    for arquivo in arquivos:
        caminho = os.path.join(base, arquivo)
        assert os.path.isfile(caminho), f"Arquivo de configuração ausente: {arquivo}"


def test_gitkeep_nas_pastas_de_dados():
    """Verifica que os arquivos .gitkeep existem nas pastas de dados ignoradas."""
    base = os.path.dirname(os.path.dirname(__file__))
    gitkeeps = [
        "data/raw/.gitkeep",
        "data/processed/.gitkeep",
        "models/trained_models/.gitkeep",
    ]
    for gitkeep in gitkeeps:
        caminho = os.path.join(base, gitkeep)
        assert os.path.isfile(caminho), f"Arquivo .gitkeep ausente: {gitkeep}"
