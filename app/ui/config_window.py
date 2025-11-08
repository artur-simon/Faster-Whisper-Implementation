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
        overlap_resolve_frame = self._create_overlap_resolve_tab(notebook)
        local_agreement_frame = self._create_local_agreement_tab(notebook)
        vad_frame = self._create_vad_tab(notebook)
        
        notebook.add(model_frame, text='Model')
        notebook.add(overlap_resolve_frame, text='Overlap res.')
        notebook.add(local_agreement_frame, text='Local agreem.')
        notebook.add(vad_frame, text='VAD')
        
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
        self.config_vars['model_size'] = (tk.StringVar(value=self.current_config.get("model_size", "medium")), str)
        model_menu = ttk.Combobox(inner_frame, textvariable=self.config_vars['model_size'][0], 
                                  values=["tiny", "base", "small", "medium", "large-v3", "turbo"],
                                  state="readonly")
        model_menu.grid(row=row, column=1, sticky='ew', pady=5)
        self.disabled_on_model_run.append(model_menu)
        row += 1
        
        ttk.Label(inner_frame, text="Device:", anchor='w').grid(row=row, column=0, sticky='w', pady=5)
        self.config_vars['device'] = (tk.StringVar(value=self.current_config.get("device", "cuda")), str)
        device_menu = ttk.Combobox(inner_frame, textvariable=self.config_vars['device'][0], 
                                   values=["cpu", "cuda"], state="readonly")
        device_menu.grid(row=row, column=1, sticky='ew', pady=5)
        self.disabled_on_model_run.append(device_menu)
        row += 1
        
        ttk.Label(inner_frame, text="Compute Type:", anchor='w').grid(row=row, column=0, sticky='w', pady=5)
        self.config_vars['compute_type'] = (tk.StringVar(value=self.current_config.get("compute_type", "int8")), str)
        compute_menu = ttk.Combobox(inner_frame, textvariable=self.config_vars['compute_type'][0], 
                                    values=["float32", "float16", "int8_float16", "int8"],
                                    state="readonly")
        compute_menu.grid(row=row, column=1, sticky='ew', pady=5)
        self.disabled_on_model_run.append(compute_menu)
        row += 1
        
        ttk.Separator(inner_frame, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky='ew', pady=10)
        row += 1
        
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
        
        self.config_vars['mic_id'] = (tk.StringVar(value=selected_mic), lambda v: self.device_index_map.get(v))
        mic_values = mic_names if mic_names else ["No devices"]
        mic_menu = ttk.Combobox(inner_frame, textvariable=self.config_vars['mic_id'][0], 
                               values=mic_values, state="readonly")
        mic_menu.grid(row=row, column=1, sticky='ew', pady=5)
        self.disabled_on_model_run.append(mic_menu)
        row += 1
        
        ttk.Label(inner_frame, text="Sample Rate:", anchor='w').grid(row=row, column=0, sticky='w', pady=5)
        self.config_vars['sample_rate'] = (tk.StringVar(value=str(self.current_config.get("sample_rate", 44100))), int)
        sample_rate_menu = ttk.Combobox(inner_frame, textvariable=self.config_vars['sample_rate'][0], 
                                        values=["16000", "22050", "44100", "48000"],
                                        state="readonly")
        sample_rate_menu.grid(row=row, column=1, sticky='ew', pady=5)
        self.disabled_on_model_run.append(sample_rate_menu)
        row += 1
        
        ttk.Label(inner_frame, text="No Speech Threshold:", anchor='w').grid(row=row, column=0, sticky='w', pady=5)
        self.config_vars['no_speech_threshold'] = (tk.StringVar(value=str(self.current_config.get("no_speech_threshold", 0.6))), float)
        threshold_entry = ttk.Entry(inner_frame, textvariable=self.config_vars['no_speech_threshold'][0])
        threshold_entry.grid(row=row, column=1, sticky='ew', pady=5)
        self.disabled_on_model_run.append(threshold_entry)
        row += 1
        
        ttk.Separator(inner_frame, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky='ew', pady=10)
        row += 1
        
        ttk.Label(inner_frame, text="Language:", anchor='w').grid(row=row, column=0, sticky='w', pady=5)
        self.config_vars['language'] = (tk.StringVar(value=self.current_config.get("language", "en")), str)
        language_menu = ttk.Combobox(inner_frame, textvariable=self.config_vars['language'][0], 
                                     values=["pt", "en"], state="readonly")
        language_menu.grid(row=row, column=1, sticky='ew', pady=5)
        row += 1
        
        ttk.Label(inner_frame, text="Algorithm:", anchor='w').grid(row=row, column=0, sticky='w', pady=5)
        self.config_vars['transcription_algorithm'] = (tk.StringVar(
            value=self.current_config.get("transcription_algorithm", "simple_overlap_resolve")
        ), str)
        algorithm_menu = ttk.Combobox(inner_frame, textvariable=self.config_vars['transcription_algorithm'][0],
                                     values=["simple_overlap_resolve", "local_agreement"],
                                     state="readonly")
        algorithm_menu.grid(row=row, column=1, sticky='ew', pady=5)
        row += 1
        
        inner_frame.columnconfigure(1, weight=1)
        return frame

    def _create_overlap_resolve_tab(self, parent):
        frame = ttk.Frame(parent)
        inner_frame = ttk.Frame(frame)
        inner_frame.pack(fill='both', expand=True, padx=20, pady=20)
        row = 0
        
        ttk.Label(inner_frame, text="Chunk Duration (s):", anchor='w').grid(row=row, column=0, sticky='w', pady=5)
        self.config_vars['chunk_duration'] = (tk.StringVar(value=str(self.current_config.get("chunk_duration", 5.0))), float)
        chunk_entry = ttk.Entry(inner_frame, textvariable=self.config_vars['chunk_duration'][0])
        chunk_entry.grid(row=row, column=1, sticky='ew', pady=5)
        self.disabled_on_model_run.append(chunk_entry)
        row += 1
        
        ttk.Label(inner_frame, text="Overlap Duration (s):", anchor='w').grid(row=row, column=0, sticky='w', pady=5)
        self.config_vars['overlap_duration'] = (tk.StringVar(value=str(self.current_config.get("overlap_duration", 1.0))), float)
        overlap_entry = ttk.Entry(inner_frame, textvariable=self.config_vars['overlap_duration'][0])
        overlap_entry.grid(row=row, column=1, sticky='ew', pady=5)
        self.disabled_on_model_run.append(overlap_entry)
        row += 1
        
        self.config_vars['use_previous_context'] = (tk.BooleanVar(value=self.current_config.get("use_previous_context", True)), bool)
        previous_context_check = ttk.Checkbutton(inner_frame, text="Use accumulated text as context", 
                                   variable=self.config_vars['use_previous_context'])
        previous_context_check.grid(row=row, column=0, columnspan=2, sticky='w', pady=5)
        row += 1
        
        inner_frame.columnconfigure(1, weight=1)
        return frame
    
    def _create_local_agreement_tab(self, parent):
        frame = ttk.Frame(parent)
        inner_frame = ttk.Frame(frame)
        inner_frame.pack(fill='both', expand=True, padx=20, pady=20)
        row = 0
        
        la_config = self.current_config.get("local_agreement_config", {})
                
        ttk.Label(inner_frame, text="Agreement Count:", anchor='w').grid(row=row, column=0, sticky='w', pady=5)
        self.config_vars['local_agreement_config.agreement_count'] = (tk.StringVar(value=str(la_config.get("agreement_count", 2))), int)
        agreement_count_entry = ttk.Entry(inner_frame, textvariable=self.config_vars['local_agreement_config.agreement_count'][0])
        agreement_count_entry.grid(row=row, column=1, sticky='ew', pady=5)
        row += 1
        
        ttk.Label(inner_frame, text="Edit Threshold:", anchor='w').grid(row=row, column=0, sticky='w', pady=5)
        self.config_vars['local_agreement_config.edit_threshold'] = (tk.StringVar(value=str(la_config.get("edit_threshold", 0.2))), float)
        edit_threshold_entry = ttk.Entry(inner_frame, textvariable=self.config_vars['local_agreement_config.edit_threshold'][0])
        edit_threshold_entry.grid(row=row, column=1, sticky='ew', pady=5)
        row += 1
        
        ttk.Label(inner_frame, text="Confidence Threshold:", anchor='w').grid(row=row, column=0, sticky='w', pady=5)
        self.config_vars['local_agreement_config.confidence_threshold'] = (tk.StringVar(value=str(la_config.get("confidence_threshold", 0.8))), float)
        confidence_threshold_entry = ttk.Entry(inner_frame, textvariable=self.config_vars['local_agreement_config.confidence_threshold'][0])
        confidence_threshold_entry.grid(row=row, column=1, sticky='ew', pady=5)
        row += 1
        
        ttk.Label(inner_frame, text="Minimum Words:", anchor='w').grid(row=row, column=0, sticky='w', pady=5)
        self.config_vars['local_agreement_config.min_words'] = (tk.StringVar(value=str(la_config.get("min_words", 3))), int)
        min_words_entry = ttk.Entry(inner_frame, textvariable=self.config_vars['local_agreement_config.min_words'][0])
        min_words_entry.grid(row=row, column=1, sticky='ew', pady=5)
        row += 1
        
        ttk.Label(inner_frame, text="Context Buffer Size:", anchor='w').grid(row=row, column=0, sticky='w', pady=5)
        self.config_vars['local_agreement_config.context_size'] = (tk.StringVar(value=str(la_config.get("context_size", 100))), int)
        context_size_entry = ttk.Entry(inner_frame, textvariable=self.config_vars['local_agreement_config.context_size'][0])
        context_size_entry.grid(row=row, column=1, sticky='ew', pady=5)
        row += 1
        
        inner_frame.columnconfigure(1, weight=1)
        return frame
      
    def _create_vad_tab(self, parent):
        frame = ttk.Frame(parent)
        inner_frame = ttk.Frame(frame)
        inner_frame.pack(fill='both', expand=True, padx=20, pady=20)
        row = 0
        
        vad_config = self.current_config.get("vad_params", {})
        
        self.config_vars['vad_filter'] = (tk.BooleanVar(value=self.current_config.get("vad_filter", True)), bool)
        vad_check = ttk.Checkbutton(inner_frame, text="Use VAD Filter", 
                                   variable=self.config_vars['vad_filter'])
        vad_check.grid(row=row, column=0, columnspan=2, sticky='w', pady=5)
        row += 1
        
        ttk.Label(inner_frame, text="threshold:", anchor='w').grid(row=row, column=0, sticky='w', pady=5)
        self.config_vars['vad_params.threshold'] = (tk.StringVar(value=str(vad_config.get("threshold", 0.5))), float)
        threshold_entry = ttk.Entry(inner_frame, textvariable=self.config_vars['vad_params.threshold'][0])
        threshold_entry.grid(row=row, column=1, sticky='ew', pady=5)
        row += 1
        
        ttk.Label(inner_frame, text="min_speech_duration_ms:", anchor='w').grid(row=row, column=0, sticky='w', pady=5)
        self.config_vars['vad_params.min_speech_duration_ms'] = (tk.StringVar(value=str(vad_config.get("min_speech_duration_ms", 400))), int)
        min_speech_duration_ms_entry = ttk.Entry(inner_frame, textvariable=self.config_vars['vad_params.min_speech_duration_ms'][0])
        min_speech_duration_ms_entry.grid(row=row, column=1, sticky='ew', pady=5)
        row += 1
        
        ttk.Label(inner_frame, text="min_silence_duration_ms:", anchor='w').grid(row=row, column=0, sticky='w', pady=5)
        self.config_vars['vad_params.min_silence_duration_ms'] = (tk.StringVar(value=str(vad_config.get("min_silence_duration_ms", 400))), int)
        min_silence_duration_ms_entry = ttk.Entry(inner_frame, textvariable=self.config_vars['vad_params.min_silence_duration_ms'][0])
        min_silence_duration_ms_entry.grid(row=row, column=1, sticky='ew', pady=5)
        row += 1
        
        inner_frame.columnconfigure(1, weight=1)
        return frame
     
    def _get_config_dict(self):
        def assign_nested(d, path, value):
            key = path[0]
            if len(path) == 1:
                d[key] = value
            else:
                d.setdefault(key, {})
                assign_nested(d[key], path[1:], value)

        config = {}
        for key, (var, caster) in self.config_vars.items():
            value = caster(var.get())
            assign_nested(config, key.split("."), value)

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