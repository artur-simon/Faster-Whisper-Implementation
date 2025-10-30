import threading
import logging
from PIL import Image, ImageDraw
import pystray
import tkinter as tk

logger = logging.getLogger("app.ui.tray_icon_manager")


class TrayIconManager:
    def __init__(self, show_window_callback, exit_callback):
        self.show_window_callback = show_window_callback
        self.exit_callback = exit_callback
        self.tray_icon = None
        self._setup()

    def _setup(self):
        image = self._get_tray_icon("OFF")
        
        self.tray_icon = pystray.Icon("WispLive", image, "WispLive", menu=pystray.Menu(
            pystray.MenuItem("Restaurar", self.show_window_callback),
            pystray.MenuItem("Sair", self.exit_callback)
        ))

        threading.Thread(target=self.tray_icon.run, name="TrayIconManager - tray_icon_run", daemon=True).start()

    def _get_tray_icon(self, state):
        image = Image.new('RGBA', (64, 64), color=(0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        
        if state == "READY":
            draw.ellipse((16, 16, 48, 48), fill=(0, 255, 0, 255))
        elif state == "RECORDING":
            draw.ellipse((16, 16, 48, 48), fill=(255, 0, 0, 255))
        else:
            draw.ellipse((16, 16, 48, 48), fill=(0, 0, 255, 255))
        
        return image

    def update_state(self, state):
        if self.tray_icon:
            self.tray_icon.icon = self._get_tray_icon(state)

    def stop(self):
        if self.tray_icon:
            self.tray_icon.stop()

