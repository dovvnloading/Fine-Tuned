import sys
import os
import torch
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QTextEdit, QFileDialog,
    QComboBox, QSpinBox, QFormLayout, QGroupBox, QCheckBox
)
from PySide6.QtCore import QObject, QThread, Signal, Slot, Qt
from PySide6.QtGui import QIcon, QPixmap

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# (STYLESHEET and MplChartWidget are unchanged)
DARK_STYLESHEET = """
QWidget {
    background-color: #2b2b2b;
    color: #f0f0f0;
    font-family: Segoe UI, Arial, sans-serif;
    font-size: 12px;
}
QMainWindow {
    background-color: #2b2b2b;
}
QGroupBox {
    border: 1px solid #444;
    border-radius: 6px;
    margin-top: 20px;
    padding: 10px 5px 5px 5px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 3px 5px 0 5px;
    background-color: #2b2b2b;
    color: #00aaff;
}
QPushButton {
    background-color: #007acc;
    color: white;
    border: 1px solid #005c99;
    padding: 5px 10px;
    border-radius: 4px;
    min-width: 10px;
}
QPushButton.SpinboxButton {
    padding: 4px 8px;
    font-weight: bold;
}
QPushButton#ChartToggleButton {
    background-color: #4a4a4a;
    text-align: left;
    padding-left: 10px;
    font-weight: bold;
}
QPushButton#ChartToggleButton:hover {
    background-color: #5a5a5a;
}
QPushButton:hover {
    background-color: #005c99;
}
QPushButton:pressed {
    background-color: #004c80;
}
QPushButton:disabled {
    background-color: #555;
    border-color: #666;
    color: #999;
}
QLineEdit, QTextEdit, QSpinBox, QComboBox {
    background-color: #3c3c3c;
    border: 1px solid #555;
    border-radius: 4px;
    padding: 4px;
}
QSpinBox {
    padding-right: 2px;
}
QSpinBox::up-button, QSpinBox::down-button {
    border: none;
    background: transparent;
    width: 0px;
    padding: 0;
    margin: 0;
}
QTextEdit {
    font-family: Consolas, Courier New, monospace;
}
QLabel {
    padding: 1px;
}
QCheckBox {
    spacing: 5px;
}
QGroupBox QCheckBox {
    margin-left: 8px;
}
QCheckBox::indicator {
    width: 12px;
    height: 12px;
}
QScrollBar:vertical {
    border: none;
    background: #3c3c3c;
    width: 10px;
    margin: 0px 0px 0px 0px;
}
QScrollBar::handle:vertical {
    background: #4a4a4a;
    min-height: 20px;
    border-radius: 5px;
}
QScrollBar::handle:vertical:hover {
    background: #5a5a5a;
}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    background: none;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
    background: none;
    border: none;
}
QScrollBar:horizontal {
    border: none;
    background: #3c3c3c;
    height: 10px;
    margin: 0px 0px 0px 0px;
}
QScrollBar::handle:horizontal {
    background: #4a4a4a;
    min-width: 20px;
    border-radius: 5px;
}
QScrollBar::handle:horizontal:hover {
    background: #5a5a5a;
}
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
    background: none;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
    background: none;
    border: none;
}
"""

class MplChartWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(5, 3), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.figure.patch.set_facecolor('#2b2b2b')
        self.ax = self.figure.add_subplot(111)
        self.setup_axes()
        layout = QVBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def setup_axes(self):
        self.ax.set_facecolor('#2b2b2b')
        self.ax.spines['bottom'].set_color('#f0f0f0')
        self.ax.spines['top'].set_color('#2b2b2b') 
        self.ax.spines['right'].set_color('#2b2b2b')
        self.ax.spines['left'].set_color('#f0f0f0')
        self.ax.tick_params(axis='x', colors='#f0f0f0')
        self.ax.tick_params(axis='y', colors='#f0f0f0')
        self.ax.yaxis.label.set_color('#f0f0f0')
        self.ax.xaxis.label.set_color('#f0f0f0')
        self.ax.title.set_color('#f0f0f0')
        self.ax.set_xlabel("Training Steps")
        self.ax.set_ylabel("Loss")
        self.ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='#555')

    def update_plot(self, steps, losses, val_steps=None, val_losses=None):
        self.ax.cla()
        if steps and losses:
            self.ax.plot(steps, losses, marker='o', linestyle='-', color='#00aaff', markersize=3, label='Training Loss')
        if val_steps and val_losses:
            self.ax.plot(val_steps, val_losses, marker='x', linestyle='--', color='#ffaf00', markersize=4, label='Validation Loss')
        if (steps and losses) or (val_steps and val_losses):
            self.ax.legend(facecolor='#3c3c3c', edgecolor='#555', labelcolor='#f0f0f0')
        self.setup_axes()
        self.canvas.draw()
        
    def clear_plot(self):
        self.ax.cla()
        self.setup_axes()
        self.canvas.draw()

class ModelLoaderWorker(QObject):
    finished = Signal(object, object); error = Signal(str)
    def __init__(self, model_name):
        super().__init__(); self.model_name = model_name
    @Slot()
    def run(self):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            model = AutoModelForCausalLM.from_pretrained(self.model_name)
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.finished.emit(model, tokenizer)
        except Exception as e:
            self.error.emit(f"Failed to load model '{self.model_name}': {e}")


class TrainingWorker(QObject):
    progress = Signal(str)
    log_data_updated = Signal(dict)
    finished = Signal(bool, object, object) # success, model, tokenizer
    error = Signal(str)

    def __init__(self, params):
        super().__init__(); self.params = params

    @Slot()
    def run(self):
        model = None
        tokenizer = None
        try:
            from datasets import load_dataset
            from transformers import (
                AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer,
                DataCollatorForLanguageModeling, TrainerCallback, EarlyStoppingCallback
            )

            class GuiLogCallback(TrainerCallback):
                def __init__(self, worker): self.worker = worker
                def on_log(self, args, state, control, logs=None, **kwargs):
                    if state.is_world_process_zero:
                        # Format the log message for better readability
                        if 'eval_loss' in logs:
                            # It's an evaluation log
                            eval_message = f"<b>&gt; Eval | Step: {state.global_step} | Eval Loss: {logs['eval_loss']:.4f}</b>"
                            self.worker.progress.emit(eval_message)
                        elif 'loss' in logs:
                            # It's a training step log
                            log_parts = []
                            log_parts.append(f"Loss: {logs['loss']:.4f}")
                            if 'learning_rate' in logs:
                                log_parts.append(f"LR: {logs['learning_rate']:.2e}")
                            if 'epoch' in logs:
                                log_parts.append(f"Epoch: {logs['epoch']:.2f}")
                            message = f"Step: {state.global_step} | " + " | ".join(log_parts)
                            self.worker.progress.emit(message)
                        
                        # Still emit the raw data for the chart
                        data = {'step': state.global_step}
                        if 'loss' in logs: data['loss'] = logs['loss']
                        if 'eval_loss' in logs: data['eval_loss'] = logs['eval_loss']
                        self.worker.log_data_updated.emit(data)
            
            self.progress.emit("--- Starting Training ---")
            tokenizer = AutoTokenizer.from_pretrained(self.params['model_name'])
            tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(self.params['model_name'])
            
            self.progress.emit("Loading and preparing dataset...")
            full_dataset = load_dataset('json', data_files=self.params['dataset_path'], split='train')
            
            train_dataset, eval_dataset = full_dataset, None
            if self.params['early_stopping_enabled']:
                self.progress.emit(f"Splitting dataset: {100-self.params['validation_size']}% train, {self.params['validation_size']}% validation...")
                split = full_dataset.train_test_split(test_size=self.params['validation_size']/100.0)
                train_dataset, eval_dataset = split['train'], split['test']

            def format_prompt(example): return {"text": f"Question: {example['question']}\nAnswer: {example['answer']}{tokenizer.eos_token}"}
            
            if self.params['use_dynamic_padding']:
                def tokenize_function(examples): return tokenizer(examples['text'], truncation=True, max_length=512)
            else:
                def tokenize_function(examples): return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)
            
            tokenized_train_dataset = train_dataset.map(format_prompt).map(tokenize_function, batched=True, remove_columns=train_dataset.column_names)
            tokenized_eval_dataset = None
            if eval_dataset:
                tokenized_eval_dataset = eval_dataset.map(format_prompt).map(tokenize_function, batched=True, remove_columns=eval_dataset.column_names)

            data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
            
            fp16_enabled, bf16_enabled = False, False
            if torch.cuda.is_available():
                if self.params['use_bf16'] and torch.cuda.is_bf16_supported():
                    bf16_enabled = True; self.progress.emit("Using BF16 mixed precision.")
                else:
                    fp16_enabled = True; self.progress.emit("Using FP16 mixed precision.")
            
            args_dict = {
                'output_dir': self.params['output_dir'],
                'num_train_epochs': self.params['epochs'],
                'per_device_train_batch_size': self.params['batch_size'],
                'gradient_accumulation_steps': self.params['gradient_accumulation_steps'],
                'learning_rate': self.params['learning_rate'], 'weight_decay': 0.01,
                'max_grad_norm': self.params['max_grad_norm'], 'warmup_steps': self.params['warmup_steps'],
                'lr_scheduler_type': self.params['lr_scheduler_type'],
                'logging_dir': f"{self.params['output_dir']}/logs",
                'logging_strategy': "steps", 'logging_steps': 10,
                'fp16': fp16_enabled, 'bf16': bf16_enabled,
                'report_to': "none", 'save_total_limit': self.params['save_total_limit'],
            }
            
            callbacks = [GuiLogCallback(self)]
            if self.params['early_stopping_enabled']:
                args_dict.update({
                    'evaluation_strategy': "steps", 'save_strategy': "steps",
                    'eval_steps': self.params['eval_steps'], 'save_steps': self.params['save_steps'],
                    'load_best_model_at_end': True, 'metric_for_best_model': "eval_loss",
                    'greater_is_better': False
                })
                callbacks.append(EarlyStoppingCallback(early_stopping_patience=self.params['patience']))
            else:
                args_dict['save_strategy'] = "epoch"
            
            training_args = TrainingArguments(**args_dict)

            trainer = Trainer(
                model=model, args=training_args, train_dataset=tokenized_train_dataset,
                eval_dataset=tokenized_eval_dataset, data_collator=data_collator, callbacks=callbacks
            )
            self.progress.emit("Starting training loop..."); trainer.train()
            
            model = trainer.model 
            
            self.progress.emit("Training complete. Saving final model to disk..."); 
            trainer.save_model(self.params['output_dir'])
            tokenizer.save_pretrained(self.params['output_dir']); self.progress.emit("Model saved successfully.")
        except Exception as e: 
            self.error.emit(str(e))
            self.finished.emit(False, None, None) # Emit failure signal
        finally:
            if model and tokenizer:
                self.finished.emit(True, model, tokenizer)


class TrainingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Language Model Fine-Tuner")
        transparent_pixmap = QPixmap(16, 16)
        transparent_pixmap.fill(Qt.transparent)
        self.setWindowIcon(QIcon(transparent_pixmap))
        self.setGeometry(100, 100, 1200, 760)
        self.training_steps = []; self.training_losses = []; self.val_steps = []; self.val_losses = []
        self.inference_model = None; self.inference_tokenizer = None
        self.central_widget = QWidget(); self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget); self.main_layout.setSpacing(10); self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.setup_ui()

    def _create_spinbox_with_controls(self, spinbox: QSpinBox) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        minus_button = QPushButton("-"); minus_button.setObjectName("SpinboxButton"); minus_button.clicked.connect(spinbox.stepDown)
        plus_button = QPushButton("+"); plus_button.setObjectName("SpinboxButton"); plus_button.clicked.connect(spinbox.stepUp)
        layout.addWidget(minus_button); layout.addWidget(spinbox, 1); layout.addWidget(plus_button)
        return widget

    def setup_ui(self):
        left_column_widget = QWidget()
        left_layout = QVBoxLayout(left_column_widget)
        left_layout.setSpacing(6); left_layout.setContentsMargins(0, 0, 0, 0)

        config_group = QGroupBox("1. Configuration")
        config_layout = QFormLayout(config_group)
        config_layout.setSpacing(6); config_layout.setContentsMargins(8, 5, 8, 8)
        self.dataset_path_edit = QLineEdit(); self.dataset_path_edit.setPlaceholderText("Select dataset.json")
        self.dataset_button = QPushButton("Browse..."); self.dataset_button.clicked.connect(self.select_dataset)
        dataset_hbox = QHBoxLayout(); dataset_hbox.addWidget(self.dataset_path_edit); dataset_hbox.addWidget(self.dataset_button)
        config_layout.addRow("Dataset File:", dataset_hbox)
        self.model_combo = QComboBox(); self.model_combo.addItems(["distilgpt2", "gpt2", "gpt2-medium"])
        self.load_base_model_button = QPushButton("Load Base Model"); self.load_base_model_button.clicked.connect(self.load_base_model)
        model_hbox = QHBoxLayout(); model_hbox.addWidget(self.model_combo, 1); model_hbox.addWidget(self.load_base_model_button)
        config_layout.addRow("Base Model:", model_hbox)
        self.output_dir_edit = QLineEdit("./finetuned-model"); self.output_dir_button = QPushButton("Browse..."); self.output_dir_button.clicked.connect(self.select_output_dir)
        output_hbox = QHBoxLayout(); output_hbox.addWidget(self.output_dir_edit); output_hbox.addWidget(self.output_dir_button)
        config_layout.addRow("Output Directory:", output_hbox)
        left_layout.addWidget(config_group)

        params_group = QGroupBox("2. Training Parameters")
        params_group.setCheckable(True); params_group.setChecked(False)
        params_main_layout = QVBoxLayout(params_group)
        params_main_layout.setSpacing(6); params_main_layout.setContentsMargins(8, 5, 8, 8)
        params_form_layout = QFormLayout(); params_form_layout.setSpacing(6)
        self.epochs_spin = QSpinBox(); self.epochs_spin.setRange(1, 100); self.epochs_spin.setValue(3)
        params_form_layout.addRow("Max Epochs:", self._create_spinbox_with_controls(self.epochs_spin))
        self.batch_size_spin = QSpinBox(); self.batch_size_spin.setRange(1, 64); self.batch_size_spin.setValue(2)
        params_form_layout.addRow("Batch Size:", self._create_spinbox_with_controls(self.batch_size_spin))
        self.lr_spin = QLineEdit("5e-5"); params_form_layout.addRow("Learning Rate:", self.lr_spin)
        self.grad_accum_spin = QSpinBox(); self.grad_accum_spin.setRange(1, 32); self.grad_accum_spin.setValue(1)
        params_form_layout.addRow("Gradient Accumulation:", self._create_spinbox_with_controls(self.grad_accum_spin))
        self.max_grad_norm_spin = QLineEdit("1.0"); params_form_layout.addRow("Max Grad Norm:", self.max_grad_norm_spin)
        self.warmup_steps_spin = QSpinBox(); self.warmup_steps_spin.setRange(0, 1000); self.warmup_steps_spin.setValue(100)
        params_form_layout.addRow("Warmup Steps:", self._create_spinbox_with_controls(self.warmup_steps_spin))
        self.lr_scheduler_combo = QComboBox(); self.lr_scheduler_combo.addItems(["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]); self.lr_scheduler_combo.setCurrentText("cosine"); params_form_layout.addRow("LR Scheduler:", self.lr_scheduler_combo)
        params_main_layout.addLayout(params_form_layout)
        self.use_bf16_checkbox = QCheckBox("Use BF16 (if available)"); self.use_bf16_checkbox.setChecked(True); params_main_layout.addWidget(self.use_bf16_checkbox)
        self.dynamic_padding_checkbox = QCheckBox("Use Dynamic Padding"); self.dynamic_padding_checkbox.setChecked(True); params_main_layout.addWidget(self.dynamic_padding_checkbox)
        left_layout.addWidget(params_group)

        self.smart_group = QGroupBox("3. Smart Stopping"); self.smart_group.setCheckable(True); self.smart_group.setChecked(True)
        smart_layout = QFormLayout(self.smart_group)
        smart_layout.setSpacing(6); smart_layout.setContentsMargins(8, 5, 8, 8)
        self.smart_group.toggled.connect(self.toggle_early_stopping_widgets)
        self.val_size_label = QLabel("Validation Set Size (%):")
        self.val_size_spin = QSpinBox(); self.val_size_spin.setRange(5, 50); self.val_size_spin.setValue(15)
        smart_layout.addRow(self.val_size_label, self._create_spinbox_with_controls(self.val_size_spin))
        self.patience_label = QLabel("Patience (evals):")
        self.patience_spin = QSpinBox(); self.patience_spin.setRange(1, 20); self.patience_spin.setValue(5)
        smart_layout.addRow(self.patience_label, self._create_spinbox_with_controls(self.patience_spin))
        self.eval_steps_label = QLabel("Eval Every (steps):")
        self.eval_steps_spin = QSpinBox(); self.eval_steps_spin.setRange(10, 1000); self.eval_steps_spin.setValue(100); self.eval_steps_spin.setSingleStep(10)
        smart_layout.addRow(self.eval_steps_label, self._create_spinbox_with_controls(self.eval_steps_spin))
        self.save_steps_label = QLabel("Save Ckpt Every (steps):")
        self.save_steps_spin = QSpinBox(); self.save_steps_spin.setRange(50, 5000); self.save_steps_spin.setValue(500); self.save_steps_spin.setSingleStep(50)
        smart_layout.addRow(self.save_steps_label, self._create_spinbox_with_controls(self.save_steps_spin))
        self.save_limit_label = QLabel("Max Ckpts to Keep:")
        self.save_limit_spin = QSpinBox(); self.save_limit_spin.setRange(1, 20); self.save_limit_spin.setValue(2)
        smart_layout.addRow(self.save_limit_label, self._create_spinbox_with_controls(self.save_limit_spin))
        left_layout.addWidget(self.smart_group)
        
        self.train_button = QPushButton("Start Training"); self.train_button.clicked.connect(self.start_training)
        left_layout.addWidget(self.train_button)
        
        self.inference_group_box = QGroupBox("Test Your Model")
        inference_layout = QVBoxLayout(self.inference_group_box); inference_layout.setSpacing(6)
        self.question_input = QLineEdit(); self.question_input.setPlaceholderText("Enter a question...")
        inference_layout.addWidget(self.question_input)
        self.ask_button = QPushButton("Ask"); self.ask_button.clicked.connect(self.ask_question)
        inference_layout.addWidget(self.ask_button)
        self.answer_output = QTextEdit(); self.answer_output.setReadOnly(True)
        self.answer_output.setPlaceholderText("Load a base model or train a new one to begin testing.")
        inference_layout.addWidget(self.answer_output, 1)
        self.question_input.setEnabled(False); self.ask_button.setEnabled(False)
        left_layout.addWidget(self.inference_group_box, 1)
        
        right_column_widget = QWidget()
        right_layout = QVBoxLayout(right_column_widget)
        right_layout.setSpacing(6); right_layout.setContentsMargins(0, 0, 0, 0)
        progress_group = QGroupBox("Training Progress")
        progress_layout = QVBoxLayout(progress_group); progress_layout.setSpacing(6)
        self.chart_toggle_button = QPushButton("▼ Hide Training Chart"); self.chart_toggle_button.setObjectName("ChartToggleButton")
        self.chart_toggle_button.setCheckable(True); self.chart_toggle_button.setChecked(True)
        self.chart_toggle_button.clicked.connect(self.toggle_chart_visibility)
        progress_layout.addWidget(self.chart_toggle_button)
        self.chart_widget = MplChartWidget(); progress_layout.addWidget(self.chart_widget, 1)
        self.log_output = QTextEdit(); self.log_output.setReadOnly(True)
        self.log_output.setPlaceholderText("Training logs will appear here...")
        progress_layout.addWidget(self.log_output, 1)
        right_layout.addWidget(progress_group)

        self.main_layout.addWidget(left_column_widget, 1)
        self.main_layout.addWidget(right_column_widget, 2)
        
        self.toggle_early_stopping_widgets(self.smart_group.isChecked())

    def _log_message(self, message, color=None, is_header=False):
        if is_header:
            self.log_output.append(f"<hr><b><font color='{color or '#00aaff'}'>{message}</font></b>")
        else:
            final_message = f"<font color='{color}'>{message}</font>" if color else message
            self.log_output.append(final_message)

    def toggle_early_stopping_widgets(self, checked):
        for i in range(self.smart_group.layout().rowCount()):
            label_widget = self.smart_group.layout().itemAt(i, QFormLayout.LabelRole).widget()
            field_widget = self.smart_group.layout().itemAt(i, QFormLayout.FieldRole).widget()
            label_widget.setVisible(checked); field_widget.setVisible(checked)

    def start_training(self):
        if not self.dataset_path_edit.text() or not os.path.exists(self.dataset_path_edit.text()):
            self._log_message("Error: Please select a valid dataset file.", color='red'); return
        self.set_controls_enabled(False); self.log_output.clear(); self.training_steps.clear(); self.training_losses.clear()
        self.val_steps.clear(); self.val_losses.clear(); self.chart_widget.clear_plot()
        self._log_message("--- TRAINING INITIALIZED ---", is_header=True)
        params = {
            'model_name': self.model_combo.currentText(), 'dataset_path': self.dataset_path_edit.text(),
            'output_dir': self.output_dir_edit.text(), 'epochs': self.epochs_spin.value(),
            'batch_size': self.batch_size_spin.value(), 'learning_rate': float(self.lr_spin.text()),
            'gradient_accumulation_steps': self.grad_accum_spin.value(),
            'max_grad_norm': float(self.max_grad_norm_spin.text()),
            'warmup_steps': self.warmup_steps_spin.value(), 'lr_scheduler_type': self.lr_scheduler_combo.currentText(),
            'use_bf16': self.use_bf16_checkbox.isChecked(), 'use_dynamic_padding': self.dynamic_padding_checkbox.isChecked(),
            'early_stopping_enabled': self.smart_group.isChecked(),
            'validation_size': self.val_size_spin.value(), 'patience': self.patience_spin.value(),
            'eval_steps': self.eval_steps_spin.value(), 'save_steps': self.save_steps_spin.value(),
            'save_total_limit': self.save_limit_spin.value()
        }
        self.train_thread = QThread(); self.train_worker = TrainingWorker(params)
        self.train_worker.moveToThread(self.train_thread)
        self.train_thread.started.connect(self.train_worker.run)
        self.train_worker.finished.connect(self.on_training_finished)
        self.train_worker.error.connect(self.on_training_error); self.train_worker.progress.connect(self.update_log)
        self.train_worker.log_data_updated.connect(self.update_chart_data)
        self.train_thread.start()
    
    def set_controls_enabled(self, enabled):
        self.train_button.setEnabled(enabled); self.load_base_model_button.setEnabled(enabled)

    def load_base_model(self):
        model_name = self.model_combo.currentText()
        self.answer_output.setText(f"Loading base model '{model_name}' from Hugging Face Hub...")
        self.set_controls_enabled(False)
        self.load_thread = QThread(); self.load_worker = ModelLoaderWorker(model_name)
        self.load_worker.moveToThread(self.load_thread)
        self.load_thread.started.connect(self.load_worker.run)
        self.load_worker.finished.connect(self.on_base_model_loaded); self.load_worker.error.connect(self.on_model_load_error)
        self.load_worker.finished.connect(self.load_thread.quit); self.load_worker.finished.connect(self.load_worker.deleteLater)
        self.load_thread.finished.connect(self.load_thread.deleteLater)
        self.load_thread.start()

    @Slot(object, object)
    def on_base_model_loaded(self, model, tokenizer):
        self.inference_model = model; self.inference_tokenizer = tokenizer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); self.inference_model.to(device)
        self.answer_output.setText(f"Base model '{self.model_combo.currentText()}' loaded successfully. Ready to test.")
        self.question_input.setEnabled(True); self.ask_button.setEnabled(True); self.set_controls_enabled(True)

    @Slot(str)
    def on_model_load_error(self, error_message):
        self.answer_output.setText(f"<font color='red'>{error_message}</font>"); self.set_controls_enabled(True)

    def toggle_chart_visibility(self):
        is_visible = self.chart_widget.isVisible()
        self.chart_widget.setVisible(not is_visible)
        self.chart_toggle_button.setText("► Show Training Chart" if is_visible else "▼ Hide Training Chart")

    def select_dataset(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Dataset", "", "JSON files (*.json *.jsonl)");
        if path: self.dataset_path_edit.setText(path)

    def select_output_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory");
        if path: self.output_dir_edit.setText(path)

    @Slot(dict)
    def update_chart_data(self, log_data):
        if 'loss' in log_data: self.training_steps.append(log_data['step']); self.training_losses.append(log_data['loss'])
        if 'eval_loss' in log_data: self.val_steps.append(log_data['step']); self.val_losses.append(log_data['eval_loss'])
        self.chart_widget.update_plot(self.training_steps, self.training_losses, self.val_steps, self.val_losses)

    def update_log(self, message): 
        self._log_message(message)

    def on_training_error(self, error_message):
        self._log_message("--- TRAINING FAILED ---", color='red', is_header=True)
        self._log_message(error_message, color='red')
        self.on_training_finished(success=False, model=None, tokenizer=None)

    @Slot(bool, object, object)
    def on_training_finished(self, success, model, tokenizer):
        self.set_controls_enabled(True)
        if success and model and tokenizer:
            self._log_message("--- TRAINING COMPLETED SUCCESSFULLY ---", color='lime', is_header=True)
            self.answer_output.setText("Best model is now loaded and ready for testing.")
            self.inference_model = model
            self.inference_tokenizer = tokenizer
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.inference_model.to(device)
            self.question_input.setEnabled(True)
            self.ask_button.setEnabled(True)
        self.train_thread.quit(); self.train_thread.wait()
        
    def ask_question(self):
        question = self.question_input.text()
        if not question or not self.inference_model or not self.inference_tokenizer: return
        self.ask_button.setText("Generating..."); self.ask_button.setEnabled(False)
        try:
            prompt = f"Question: {question}\nAnswer:"
            inputs = self.inference_tokenizer(prompt, return_tensors="pt").to(self.inference_model.device)
            output = self.inference_model.generate(
                **inputs, max_new_tokens=350, num_return_sequences=1, no_repeat_ngram_size=2,
                early_stopping=True, eos_token_id=self.inference_tokenizer.eos_token_id,
                pad_token_id=self.inference_tokenizer.eos_token_id
            )
            generated_text = self.inference_tokenizer.decode(output[0], skip_special_tokens=True)
            answer = generated_text.split("Answer:")[-1].strip()
            self.answer_output.setText(answer)
        except Exception as e: self.answer_output.setText(f"<font color='red'>An error occurred during inference:\n{e}</font>")
        finally: self.ask_button.setText("Ask"); self.ask_button.setEnabled(True)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_STYLESHEET)
    window = TrainingApp()
    window.show()
    sys.exit(app.exec())