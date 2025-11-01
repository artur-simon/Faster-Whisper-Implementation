import tkinter as tk
from tkinter import ttk, messagebox
import logging
from app.audio.audio_capture import AudioCapture
from app.utils.config_manager import ConfigManager

logger = logging.getLogger("app.ui.config_window")


class ConfigWindow:
    def __init__(self, parent, config_manager:ConfigManager, current_config, is_model_running=False, theme_manager=None):
        self.parent = parent
        self.config_manager = config_manager
        self.current_config = current_config
        self.theme_manager = theme_manager
        
        self.window = tk.Toplevel(parent.root)
        self.window.title("Configuration")
        self.window.geometry("400x450")
        self.window.protocol("WM_DELETE_WINDOW", self._close)
        
        self.config_vars = {}
        self.input_devices = AudioCapture.get_input_devices()
        self.device_index_map = {f"{d['index']}: {d['name']}": d['index'] for d in self.input_devices}
        self.disabled_on_model_run = []
        
        self._create_widgets()
        self.toggle_widgets(is_model_running)
        
        if self.theme_manager:
            self.theme_manager.apply_theme(self.window)
        
    def _create_widgets(self):
        main_frame = ttk.Frame(self.window)
        main_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill='both', expand=True)
        
        model_frame = self._create_model_tab(notebook)
        audio_frame = self._create_audio_tab(notebook)
        
        notebook.add(model_frame, text='Model')
        notebook.add(audio_frame, text='Audio')
        
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill='x', pady=(5, 0))
        
        ttk.Button(button_frame, text="Apply", command=self._apply_config).pack(side='left', padx=5)
        ttk.Button(button_frame, text="OK", command=self._ok).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Cancel", command=self._close).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Save Configuration", command=self._save_config).pack(side='right', padx=5)
        
    def _create_model_tab(self, parent):
        frame = ttk.Frame(parent)
        inner_frame = ttk.Frame(frame)
        inner_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        row = 0
        
        ttk.Label(inner_frame, text="Model Size:", anchor='w').grid(row=row, column=0, sticky='w', pady=5)
        self.config_vars['model_size'] = tk.StringVar(value=self.current_config.get("model_size", "medium"))
        model_menu = ttk.Combobox(inner_frame, textvariable=self.config_vars['model_size'], 
                                  values=["tiny", "base", "small", "medium", "large-v3", "turbo"],
                                  state="readonly")
        model_menu.grid(row=row, column=1, sticky='ew', pady=5)
        self.disabled_on_model_run.append(model_menu)
        row += 1
        
        ttk.Label(inner_frame, text="Device:", anchor='w').grid(row=row, column=0, sticky='w', pady=5)
        self.config_vars['device'] = tk.StringVar(value=self.current_config.get("device", "cuda"))
        device_menu = ttk.Combobox(inner_frame, textvariable=self.config_vars['device'], 
                                   values=["cpu", "cuda"], state="readonly")
        device_menu.grid(row=row, column=1, sticky='ew', pady=5)
        self.disabled_on_model_run.append(device_menu)
        row += 1
        
        ttk.Label(inner_frame, text="Compute Type:", anchor='w').grid(row=row, column=0, sticky='w', pady=5)
        self.config_vars['compute_type'] = tk.StringVar(value=self.current_config.get("compute_type", "int8"))
        compute_menu = ttk.Combobox(inner_frame, textvariable=self.config_vars['compute_type'], 
                                    values=["float32", "float16", "int8_float16", "int8"],
                                    state="readonly")
        compute_menu.grid(row=row, column=1, sticky='ew', pady=5)
        self.disabled_on_model_run.append(compute_menu)
        row += 1
        
        inner_frame.columnconfigure(1, weight=1)
        
        return frame
        
    def _create_audio_tab(self, parent):
        frame = ttk.Frame(parent)
        inner_frame = ttk.Frame(frame)
        inner_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        row = 0
        
        ttk.Label(inner_frame, text="Audio Input:", anchor='w').grid(row=row, column=0, sticky='w', pady=5)
        
        mic_names = list(self.device_index_map.keys())
        selected_mic = "No devices"
        
        if mic_names:
            default_id = AudioCapture.get_default_input_device_id()
            selected_mic = next(
                (f"{d['index']}: {d['name']}" for d in self.input_devices if d["index"] == default_id),
                None
            )
                
        saved_mic_idx = self.current_config.get("mic_id")
        if saved_mic_idx is not None:
            selected_mic = next(
                (name for name, idx in self.device_index_map.items() if idx == saved_mic_idx),
                selected_mic
            )
        
        self.config_vars['mic_id'] = tk.StringVar(value=selected_mic)
        mic_values = mic_names if mic_names else ["No devices"]
        mic_menu = ttk.Combobox(inner_frame, textvariable=self.config_vars['mic_id'], 
                               values=mic_values, state="readonly")
        mic_menu.grid(row=row, column=1, sticky='ew', pady=5)
        self.disabled_on_model_run.append(mic_menu)
        row += 1
        
        ttk.Label(inner_frame, text="Sample Rate:", anchor='w').grid(row=row, column=0, sticky='w', pady=5)
        self.config_vars['sample_rate'] = tk.StringVar(value=str(self.current_config.get("sample_rate", 44100)))
        sample_rate_menu = ttk.Combobox(inner_frame, textvariable=self.config_vars['sample_rate'], 
                                        values=["16000", "22050", "44100", "48000"],
                                        state="readonly")
        sample_rate_menu.grid(row=row, column=1, sticky='ew', pady=5)
        self.disabled_on_model_run.append(sample_rate_menu)
        row += 1
        
        ttk.Label(inner_frame, text="Chunk Duration (s):", anchor='w').grid(row=row, column=0, sticky='w', pady=5)
        self.config_vars['chunk_duration'] = tk.StringVar(value=str(self.current_config.get("chunk_duration", 5.0)))
        chunk_entry = ttk.Entry(inner_frame, textvariable=self.config_vars['chunk_duration'])
        chunk_entry.grid(row=row, column=1, sticky='ew', pady=5)
        self.disabled_on_model_run.append(chunk_entry)
        row += 1
        
        ttk.Label(inner_frame, text="Overlap Duration (s):", anchor='w').grid(row=row, column=0, sticky='w', pady=5)
        self.config_vars['overlap_duration'] = tk.StringVar(value=str(self.current_config.get("overlap_duration", 1.0)))
        overlap_entry = ttk.Entry(inner_frame, textvariable=self.config_vars['overlap_duration'])
        overlap_entry.grid(row=row, column=1, sticky='ew', pady=5)
        self.disabled_on_model_run.append(overlap_entry)
        row += 1
        
        ttk.Label(inner_frame, text="No Speech Threshold:", anchor='w').grid(row=row, column=0, sticky='w', pady=5)
        self.config_vars['no_speech_threshold'] = tk.StringVar(value=str(self.current_config.get("no_speech_threshold", 0.6)))
        threshold_entry = ttk.Entry(inner_frame, textvariable=self.config_vars['no_speech_threshold'])
        threshold_entry.grid(row=row, column=1, sticky='ew', pady=5)
        self.disabled_on_model_run.append(threshold_entry)
        row += 1
        
        self.config_vars['vad_filter'] = tk.BooleanVar(value=self.current_config.get("vad_filter", True))
        vad_check = ttk.Checkbutton(inner_frame, text="Use VAD Filter", 
                                   variable=self.config_vars['vad_filter'])
        vad_check.grid(row=row, column=0, columnspan=2, sticky='w', pady=5)
        self.disabled_on_model_run.append(vad_check)
        row += 1
        
        self.config_vars['use_previous_context'] = tk.BooleanVar(value=self.current_config.get("use_previous_context", True))
        previous_context_check = ttk.Checkbutton(inner_frame, text="Use accumulated text as context", 
                                   variable=self.config_vars['use_previous_context'])
        previous_context_check.grid(row=row, column=0, columnspan=2, sticky='w', pady=5)
        row += 1
        
        
        ttk.Label(inner_frame, text="Language:", anchor='w').grid(row=row, column=0, sticky='w', pady=5)
        self.config_vars['language'] = tk.StringVar(value=self.current_config.get("language", "en"))
        language_menu = ttk.Combobox(inner_frame, textvariable=self.config_vars['language'], 
                                     values=["pt", "en"], state="readonly")
        language_menu.grid(row=row, column=1, sticky='ew', pady=5)
        row += 1
            
        inner_frame.columnconfigure(1, weight=1)
        
        return frame
    
    def _get_config_dict(self):
        config = {}
        for key, var in self.config_vars.items():
            value = var.get()
            if key == 'mic_id':
                config[key] = self.device_index_map.get(value)
            elif key in ['sample_rate']:
                config[key] = int(value)
            elif key in ['chunk_duration', 'overlap_duration', 'no_speech_threshold']:
                config[key] = float(value)
            else:
                config[key] = value
        return config
    
    def _apply_config(self):
        try:
            config = self._get_config_dict()
            current_config = self.config_manager.get_config_dict()
            current_config.update(config)
            logger.info(f"Applying config: {current_config}")
            
            self.parent.on_config_change(current_config)
            
            logger.info("Config applied successfully")
        except ValueError as e:
            logger.error(f"Invalid config values: {e}")
            messagebox.showerror("Error", "Invalid configuration values")
    
    def _ok(self):
        self._apply_config()
        self._close()
        
    def _close(self):
        self.window.destroy()
        self.parent.config_window = None
    
    def _save_config(self):
        try:
            config = self._get_config_dict()
            current_config = self.config_manager.get_config_dict()
            current_config.update(config)
            
            current_config["should_paste_content"] = self.parent.should_paste_content_var.get()
            
            self.config_manager.save_config_to_file(current_config)
            logger.info("Configuration saved successfully")
            messagebox.showinfo("Success", "Configuration saved!")
        except Exception as e:
            logger.error(f"Error saving config: {e}")
            messagebox.showerror("Error", "Error saving configuration")
    
    def show(self):
        self.window.deiconify()
        self.window.grab_set()

    def toggle_widgets(self, is_model_loaded):
        state = tk.DISABLED if is_model_loaded else tk.NORMAL
        for widget in self.disabled_on_model_run:
            widget.config(state=state)