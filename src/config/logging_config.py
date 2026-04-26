import logging
import sys
from datetime import UTC, datetime

from pythonjsonlogger import json


class JsonFormatter(json.JsonFormatter):
    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)
        log_record["timestamp"] = datetime.now(UTC).isoformat()
        log_record["level"] = record.levelname
        log_record["logger"] = record.name
        log_record["services"] = "churn_prediction_api"

        if not log_record.get("message"):
            log_record["message"] = record.getMessage()


def setup_api_logger():
    logger = logging.getLogger("churn_prediction_api")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = JsonFormatter()
    handler.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(handler)
    return logger


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,  # ou INFO, dependendo do nível de detalhe desejado
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )
