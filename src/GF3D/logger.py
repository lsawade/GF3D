# setup_logger.py
import logging
from .constants import LOG_LEVEL

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger('GF')