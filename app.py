from tkinter import filedialog, messagebox
from PIL import Image, ImageDraw
import tkinter as tk
import threading
import pyperclip
import pystray

from app.transcription.whisper_transcriber import LiveWhisperTranscriber
from app.ui.live_text_view import LiveTextViewer

class WhisperVoiceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Whisper Voice Transcriber")
        
        # TODO System tray
        self.root.bind("<Configure>", self.hide_window)
        self.root.protocol("WM_DELETE_WINDOW", self.exit_app)
        
        config_frame = tk.Frame(root)
        config_frame.pack(pady=5)
        
        #Model selection
        self.model_size = tk.StringVar(value="medium")
        tk.OptionMenu(config_frame, self.model_size, "tiny", "base", "small", "medium", "large-v3").grid(row=0, column=0)
        
        # Device selection
        self.device_var = tk.StringVar(value="cuda")
        tk.OptionMenu(config_frame, self.device_var, "cpu", "cuda").grid(row=0, column=1)

        # Compute Type selection
        self.compute_type_var = tk.StringVar(value="int8")
        tk.OptionMenu(config_frame, self.compute_type_var, "float32", "float16", "int8_float16", "int8").grid(row=0, column=2)
        
        # Compute Type selection
        self.language = tk.StringVar(value="pt")
        tk.OptionMenu(config_frame, self.language, "pt", "en").grid(row=0, column=3)

        # Toggle model activation
        self.toggle_model_button = tk.Button(config_frame, text="Ativar Modelo", command=self.toggle_model)
        self.toggle_model_button.grid(row=0, column=4)

        # Control buttons
        button_frame = tk.Frame(root)
        button_frame.pack(pady=5)
        
        self.start_button = tk.Button(button_frame, text="Iniciar Gravação", command=self.start_recording, state=tk.DISABLED)
        self.start_button.grid(row=0, column=0)
        self.stop_button = tk.Button(button_frame, text="Parar Gravação", command=self.stop_recording, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=1)
        
        self.copy_button = tk.Button(button_frame, text="Copiar Texto", command=self.copy_text)
        self.copy_button.grid(row=0, column=2)

        self.select_file_button = tk.Button(button_frame, text="Selecionar Arquivo de Áudio", command=self.select_audio_file)
        self.select_file_button.grid(row=0, column=3)
        
        self.status_label = tk.Label(root, text="Status: Parado")
        self.status_label.pack(pady=5)

        self.transcriber = None
        self.is_running = False
        self.tray_icon = None

        # Update transcription
        _ = LiveTextViewer(root, "transcription.txt")
        
    def toggle_model(self):
        if self.transcriber is None:
            device = self.device_var.get()
            compute_type = self.compute_type_var.get()
            model_size = self.model_size.get()
            language = self.language.get()
            
            try:
                self.transcriber = LiveWhisperTranscriber(model_size=model_size, device=device, compute_type=compute_type, language=language)
                self.toggle_model_button.config(text="Desativar Modelo")
                self.start_button.config(state=tk.NORMAL)
                self.copy_button.config(state=tk.NORMAL)
                self.select_file_button.config(state=tk.NORMAL)

                # Desabilitar opções
                for widget in [self.device_var, self.compute_type_var]:
                    widget.set(widget.get())  # força o valor atual
                for child in self.root.winfo_children():
                    if isinstance(child, tk.OptionMenu):
                        child.config(state=tk.DISABLED)
            except Exception as e:
                self.status_label.config(text=f"Erro ao iniciar modelo: {e}")
        else:
            self.transcriber.shutdown()
            self.transcriber = None
            self.toggle_model_button.config(text="Ativar Modelo")
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.DISABLED)
            self.copy_button.config(state=tk.DISABLED)
            self.select_file_button.config(state=tk.DISABLED)

            # Reabilita opções
            for child in self.root.winfo_children():
                if isinstance(child, tk.OptionMenu):
                    child.config(state=tk.NORMAL)
                
    def start_recording(self):
        if not self.is_running:
            self.is_running = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.status_label.config(text="Status: Gravando...")

            self.transcriber.running = True
            self.transcriber.run(output_file="transcription.txt")
            
    def stop_recording(self):
        if self.is_running and self.transcriber:
            self.transcriber.running = False
            self.is_running = False
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.status_label.config(text="Status: Parado")

    def select_audio_file(self):
        filetypes = (("MP3 files","*.mp3"), ("WAV files", "*.wav"), ("All files", "*.*"))
        filepath = filedialog.askopenfilename(title="Selecione um arquivo de áudio", filetypes=filetypes)
        if filepath and self.transcriber:
            self.status_label.config(text="Status: Transcrevendo arquivo...")
            threading.Thread(target=self.transcriber.transcribe_audio, args=(filepath, "transcription.txt"), name="Thread-Transcribe-File").start()

    def copy_text(self):
        content = self.text_area.get(1.0, tk.END)
        pyperclip.copy(content.strip())
        messagebox.showinfo("Copiado", "Transcrição copiada para a área de transferência!")

    def setup_tray_icon(self):
        image = Image.new('RGB', (64, 64), color=(0, 0, 255))
        draw = ImageDraw.Draw(image)
        draw.ellipse((16, 16, 48, 48), fill=(255, 255, 255))

        self.tray_icon = pystray.Icon("Whisper Voice", image, "Whisper Voice", menu=pystray.Menu(
            pystray.MenuItem("Restaurar", self.show_window),
            pystray.MenuItem("Sair", self.exit_app)
        ))

        threading.Thread(target=self.tray_icon.run, daemon=True).start()

    def hide_window(self, event):
        if root.state() == 'iconic':
            self.root.withdraw()
            self.setup_tray_icon()

    def show_window(self, icon=None, item=None):
        self.root.deiconify()
        if self.tray_icon:
            self.tray_icon.stop()

    def exit_app(self, icon=None, item=None):
        if self.tray_icon:
            self.tray_icon.stop()
            
        if self.transcriber:
            self.transcriber.shutdown()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = WhisperVoiceApp(root)
    root.mainloop()