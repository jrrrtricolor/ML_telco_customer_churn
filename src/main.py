import logging
from pathlib import Path

from src.models.EDA import EDA
from src.models.treino import Treino
from src.utils.arquivo import Arquivo
from src.train import train
from src.preprocess import preprocess

if __name__ == "__main__":
    preprocess()
    train()