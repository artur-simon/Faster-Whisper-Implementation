import pytest
import logging
import queue
from app.utils.logging_manager import LoggingManager, QueueHandler


class TestQueueHandler:
    def test_handler_initialization(self):
        log_queue = queue.Queue()
        handler = QueueHandler(log_queue)
        
        assert handler.log_queue == log_queue
    
    def test_emit_adds_to_queue(self):
        log_queue = queue.Queue()
        handler = QueueHandler(log_queue)
        
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="test message",
            args=(),
            exc_info=None
        )
        
        handler.emit(record)
        assert log_queue.qsize() == 1
        
        retrieved_record = log_queue.get_nowait()
        assert retrieved_record.getMessage() == "test message"
    
    def test_emit_handles_full_queue(self):
        log_queue = queue.Queue(maxsize=1)
        handler = QueueHandler(log_queue)
        
        record1 = logging.LogRecord(
            name="test", level=logging.INFO, pathname="test.py",
            lineno=1, msg="message 1", args=(), exc_info=None
        )
        record2 = logging.LogRecord(
            name="test", level=logging.INFO, pathname="test.py",
            lineno=1, msg="message 2", args=(), exc_info=None
        )
        
        handler.emit(record1)
        handler.emit(record2)
        
        assert log_queue.qsize() == 1


class TestLoggingManager:
    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        LoggingManager._instance = None
        yield
        LoggingManager._instance = None
    
    def test_singleton_pattern(self):
        manager1 = LoggingManager.get_instance()
        manager2 = LoggingManager.get_instance()
        
        assert manager1 is manager2
    
    def test_double_initialization_raises_error(self):
        LoggingManager()
        
        with pytest.raises(RuntimeError, match="LoggingManager is a singleton"):
            LoggingManager()
    
    def test_get_logger_adds_prefix(self):
        manager = LoggingManager()
        logger = manager.get_logger("test_module")
        
        assert logger.name == "app.test_module"
    
    def test_get_logger_preserves_app_prefix(self):
        manager = LoggingManager()
        logger = manager.get_logger("app.test_module")
        
        assert logger.name == "app.test_module"
    
    def test_set_log_level(self):
        manager = LoggingManager()
        manager.set_log_level(logging.WARNING)
        
        app_logger = logging.getLogger("app")
        assert app_logger.level == logging.WARNING
    
    def test_clear_queue(self):
        manager = LoggingManager()
        
        for i in range(5):
            manager.log_queue.put(f"message {i}")
        
        manager.clear_queue()
        assert manager.log_queue.empty()
    
    def test_app_logger_configured(self):
        manager = LoggingManager()
        app_logger = logging.getLogger("app")
        
        assert app_logger.level == logging.DEBUG
        assert manager.queue_handler in app_logger.handlers
    
    def test_logging_to_queue(self):
        manager = LoggingManager()
        logger = manager.get_logger("test")
        
        logger.info("test message")
        
        assert not manager.log_queue.empty()
        record = manager.log_queue.get_nowait()
        assert "test message" in record.getMessage()
    
    def test_queue_handler_formatter(self):
        manager = LoggingManager()
        
        formatter = manager.queue_handler.formatter
        assert formatter is not None
        assert formatter.datefmt == '%H:%M:%S'


