import logging
import queue
from typing import Optional, Callable


class QueueHandler(logging.Handler):
    def __init__(self, log_queue: queue.Queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        try:
            self.log_queue.put_nowait(record)
        except queue.Full:
            pass


class LoggingManager:
    _instance: Optional['LoggingManager'] = None
    
    def __init__(self):
        if LoggingManager._instance is not None:
            raise RuntimeError("LoggingManager is a singleton")
        
        self.log_queue = queue.Queue(maxsize=1000)
        self.queue_handler = QueueHandler(self.log_queue)
        self.queue_handler.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        self.queue_handler.setFormatter(formatter)
        
        self._setup_loggers()
        LoggingManager._instance = self
    
    @classmethod
    def get_instance(cls) -> 'LoggingManager':
        if cls._instance is None:
            cls._instance = LoggingManager()
        return cls._instance
    
    def _setup_loggers(self):
        app_logger = logging.getLogger("app")
        app_logger.setLevel(logging.DEBUG)
        app_logger.addHandler(self.queue_handler)
        
        # faster_whisper_logger = logging.getLogger("faster_whisper")
        # faster_whisper_logger.setLevel(logging.DEBUG)
        # faster_whisper_logger.addHandler(self.queue_handler)
    
    def set_log_level(self, level: int):
        logging.getLogger("app").setLevel(level)
    
    def set_faster_whisper_level(self, level: int):
        logging.getLogger("faster_whisper").setLevel(level)
    
    def get_logger(self, name: str) -> logging.Logger:
        if not name.startswith("app."):
            name = f"app.{name}"
        return logging.getLogger(name)
    
    def clear_queue(self):
        while not self.log_queue.empty():
            try:
                self.log_queue.get_nowait()
            except queue.Empty:
                break

