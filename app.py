import tkinter as tk
import logging
from app.ui.main_window import MainWindow
from app.utils.logging_manager import LoggingManager


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    logging_manager = LoggingManager()
    logger = logging_manager.get_logger("main")
    logger.info("Starting WispLive application")

    root = tk.Tk()
    app = MainWindow(root)
    root.mainloop()
