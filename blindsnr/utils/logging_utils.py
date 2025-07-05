import logging
import sys

# Configure package-wide logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

# Create console handler
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.WARNING)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Make logger available when importing package
__all__ = ['logger']