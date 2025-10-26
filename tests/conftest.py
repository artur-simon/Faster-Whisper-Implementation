import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@pytest.fixture(scope="session")
def fixtures_dir():
    return os.path.join(os.path.dirname(__file__), "fixtures")


@pytest.fixture(autouse=True)
def reset_logging():
    import logging
    from app.utils.logging_manager import LoggingManager
    
    LoggingManager._instance = None
    
    yield
    
    LoggingManager._instance = None
    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)


