import sys
import tkinter as tk
import logging
from app.ui.main_window import MainWindow
from app.utils.logging_manager import LoggingManager


if __name__ == "__main__":
    # When frozen, the Studio re-launches this same exe with ``--studio <folder>``
    # (it can't use ``-m`` against the bootloader). Dispatch to the Qt Studio.
    if len(sys.argv) >= 3 and sys.argv[1] == "--studio":
        from app.ui.qt.transcription_studio import main as studio_main

        raise SystemExit(studio_main([sys.argv[0], sys.argv[2]]))

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
