import os
import tkinter as tk
import logging

logger = logging.getLogger("app.ui.live_text_view")


class LiveTextViewer:
    def __init__(self, root, path):
        self.root = root
        self.path = path
        
        frame = tk.Frame(root)
        frame.pack(fill='both', expand=True)

        self.scrollbar = tk.Scrollbar(frame)
        self.scrollbar.pack(side='right', fill='y')

        self.text = tk.Text(frame, wrap='word', yscrollcommand=self.scrollbar.set)
        self.text.pack(side='left', fill='both', expand=True)
        self.scrollbar.config(command=self.text.yview)

        self.last_size = 0
        self.follow = True
        self.text.bind('<MouseWheel>', self._on_scroll)
        self.update()

    def _on_scroll(self, event):
        last = self.text.index("@0,10000")
        total = self.text.index("end-1c")
        self.follow = last.split(".")[0] == total.split(".")[0]

    def update(self):
        try:
            if(os.path.exists(self.path)):
                size = os.path.getsize(self.path)
                if size < self.last_size:
                    self.text.delete("1.0", tk.END)
                    self.last_size = 0
                if size > self.last_size:
                    with open(self.path, "r", encoding="utf-8") as f:
                        f.seek(self.last_size)
                        new_data = f.read()
                    self.text.insert(tk.END, new_data)
                    self.last_size = size
                    if self.follow:
                        self.text.see(tk.END)
        except Exception as e:
            logger.error(f"Error updating live text view: {e}", exc_info=True)
        self.root.after(500, self.update)
