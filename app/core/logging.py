import json
import logging
from datetime import datetime, timezone


_REDACTED_KEYS = {"api_key", "openai_api_key", "content", "document_content"}

# Internal LogRecord attrs we don't want to re-emit
_LOGRECORD_ATTRS = frozenset(logging.LogRecord(
    "", 0, "", 0, "", (), None
).__dict__.keys()) | {"message", "asctime"}


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, val in record.__dict__.items():
            if key in _LOGRECORD_ATTRS or key in _REDACTED_KEYS or key.startswith("_"):
                continue
            payload[key] = val
        return json.dumps(payload)


def get_logger(name: str) -> logging.Logger:
    from app.core.config import settings

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(_JsonFormatter())
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
    return logger
