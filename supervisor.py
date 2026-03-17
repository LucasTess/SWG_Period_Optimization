import sys
import numpy as np
import copy
import datetime
import pandas as pd
import os
import json
import traceback
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QCheckBox, 
                             QPushButton, QRadioButton, QGroupBox, QFormLayout, 
                             QDoubleSpinBox, QSpinBox, QMessageBox, QTextEdit)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# Importa a função trabalhadora de main.py
try:
    from main import run_optimization
except ImportError:
    print("Erro: Certifique-se de que main.py está no mesmo diretório.")

STATE_FILE = "supervisor_state.json"

# --- DEFINIÇÃO DA CONFIGURAÇÃO PADRÃO ---
DEFAULT_CONFIG = {
    "file_paths": {
        "original_lms_file_name": "SWG_period_EME.lms",
        "geometry_lsf_script_name": "create_guide_EME.lsf",
        "simulation_lsf_script_name": "run_simu_guide_EME.lsf",
        "simulation_results_directory_name": "simulation_results"
    },
    "ga_params": {}, # Preenchido via UI
    "ga_ranges": {}, # Preenchido via UI
    "fitness_params": {
        "strategy_name": "reflection_band",
        "cutoff_wl_nm": 1550,
        "center_wl_nm": 1550,
        "bandwidth_nm": 5,
        "transition_bw_nm": 20,
        "weights": {"rejection": 0.50, "passband": 0.20, "transition": 0.30}
    },
    "run_settings": {
        "clean_temp_files": True,
        "lumerical_hide_ui": True
    }
}

class OptimizationWorker(QThread):
    progress_sig = pyqtSignal(str)
    finished_sig = pyqtSignal(object)
    
    # Flag estática para que o main.py possa consultar a parada
    _STOP_REQUESTED = False

    def __init__(self, config_base, sweep_params, start_index=0):
        super().__init__()
        self.config_base = config_base
        self.sweep_params = sweep_params
        self.start_index = start_index
        OptimizationWorker._STOP_REQUESTED = False

    @staticmethod
    def should_stop():
        return OptimizationWorker._STOP_REQUESTED

    def stop(self):
        OptimizationWorker._STOP_REQUESTED = True

    def run(self):
        try:
            results = self.execute_logic()
            self.finished_sig.emit(results)
        except Exception:
            self.finished_sig.emit(traceback.format_exc())

    def execute_logic(self):
        if not self.sweep_params['enabled']:
            targets = [self.sweep_params['start']]
        else:
            targets = np.linspace(self.sweep_params['start'], 
                                 self.sweep_params['stop'], 
                                 self.sweep_params['steps'])
        
        summary = []
        for i in range(self.start_index, len(targets)):
            if OptimizationWorker.should_stop(): return "Otimização interrompida."
            
            target_wl = targets[i]
            self.progress_sig.emit(f"--- Etapa {i+1}/{len(targets)}: Alvo {target_wl:.2f} nm ---")
            
            for bw in [5, 10, 20]:
                if OptimizationWorker.should_stop(): return "Parada solicitada."
                
                cfg = copy.deepcopy(self.config_base)
                cfg['fitness_params']['center_wl_nm'] = float(target_wl)
                cfg['fitness_params']['bandwidth_nm'] = float(bw)
                
                self.progress_sig.emit(f"  -> Tentando BW: {bw} nm")
                
                # Chama a otimização passando a flag de checagem
                best_fit, _ = run_optimization(cfg, stop_check=OptimizationWorker.should_stop)
                
                self.save_current_state(i + 1)

                if best_fit >= 0.75:
                    self.progress_sig.emit(f"  [Sucesso] Fitness {best_fit:.4f} atingido.")
                    summary.append({"wl": target_wl, "bw": bw, "fit": best_fit})
                    break
        return summary

    def save_current_state(self, next_index):
        state = {
            "next_index": next_index,
            "timestamp": datetime.datetime.now().isoformat(),
            "config_base": self.config_base,
            "sweep_params": self.sweep_params
        }
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=4)

class BraggSupervisorUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bragg Guide Optimization Supervisor")
        self.setMinimumWidth(700)
        self.worker = None
        self.setup_ui()

    def setup_ui(self):
        container = QWidget()
        self.setCentralWidget(container)
        layout = QVBoxLayout(container)

        # 1. GA Settings
        ga_box = QGroupBox("Genetic Algorithm Settings")
        ga_form = QFormLayout()
        self.pop_size = QSpinBox(); self.pop_size.setRange(2, 1000); self.pop_size.setValue(50)
        self.mut_rate = QDoubleSpinBox(); self.mut_rate.setRange(0, 1); self.mut_rate.setValue(0.2)
        self.num_gen = QSpinBox(); self.num_gen.setRange(1, 2000); self.num_gen.setValue(160)
        self.conv_check = QCheckBox("Enable Convergence Check"); self.conv_check.setChecked(True)
        self.p_ratio = QDoubleSpinBox(); self.p_ratio.setRange(0, 1); self.p_ratio.setValue(0.2)
        self.min_p = QSpinBox(); self.min_p.setRange(1, 500); self.min_p.setValue(20)

        ga_form.addRow("Population Size:", self.pop_size)
        ga_form.addRow("Mutation Rate:", self.mut_rate)
        ga_form.addRow("Total Generations:", self.num_gen)
        ga_form.addRow(self.conv_check)
        ga_form.addRow("Patience Ratio (%):", self.p_ratio)
        ga_form.addRow("Min Patience (Gens):", self.min_p)
        ga_box.setLayout(ga_form)
        layout.addWidget(ga_box)

        # 2. Experiment Mode
        type_box = QGroupBox("Experiment Mode")
        type_layout = QHBoxLayout()
        self.rad_uniform = QRadioButton("Uniform Guide"); self.rad_apodized = QRadioButton("Apodized Guide")
        self.rad_uniform.setChecked(True); self.rad_uniform.toggled.connect(self.refresh_range_fields)
        type_layout.addWidget(self.rad_uniform); type_layout.addWidget(self.rad_apodized)
        type_box.setLayout(type_layout)
        layout.addWidget(type_box)

        # 3. Ranges
        self.range_box = QGroupBox("Optimization Parameter Ranges")
        self.range_form = QFormLayout(); self.range_box.setLayout(self.range_form)
        layout.addWidget(self.range_box); self.refresh_range_fields()

        # 4. Wavelength Control
        sweep_box = QGroupBox("Target Wavelength Control")
        sweep_form = QFormLayout()
        self.chk_sweep = QCheckBox("Enable Wavelength Sweep"); self.chk_sweep.setChecked(True)
        self.chk_sweep.toggled.connect(self.toggle_sweep_ui)
        self.wl_start = QDoubleSpinBox(); self.wl_start.setRange(1000, 2000); self.wl_start.setValue(1450)
        self.wl_stop = QDoubleSpinBox(); self.wl_stop.setRange(1000, 2000); self.wl_stop.setValue(1550)
        self.wl_steps = QSpinBox(); self.wl_steps.setRange(1, 100); self.wl_steps.setValue(4)
        sweep_form.addRow(self.chk_sweep); sweep_form.addRow("Start WL (nm):", self.wl_start)
        sweep_form.addRow("Stop WL (nm):", self.wl_stop); sweep_form.addRow("Steps:", self.wl_steps)
        sweep_box.setLayout(sweep_form); layout.addWidget(sweep_box)

        # 5. Control Buttons
        btn_layout = QHBoxLayout()
        self.btn_start = QPushButton("START NEW"); self.btn_start.setStyleSheet("height: 40px; background-color: #2E7D32; color: white;")
        self.btn_resume = QPushButton("RESUME"); self.btn_resume.setStyleSheet("height: 40px; background-color: #1565C0; color: white;")
        self.btn_stop = QPushButton("STOP"); self.btn_stop.setStyleSheet("height: 40px; background-color: #C62828; color: white;"); self.btn_stop.setEnabled(False)
        
        self.btn_start.clicked.connect(lambda: self.start_opt(restart=True))
        self.btn_resume.clicked.connect(lambda: self.start_opt(restart=False))
        self.btn_stop.clicked.connect(self.stop_opt)
        
        btn_layout.addWidget(self.btn_start); btn_layout.addWidget(self.btn_resume); btn_layout.addWidget(self.btn_stop)
        layout.addLayout(btn_layout)

        self.log_output = QTextEdit(); self.log_output.setReadOnly(True); layout.addWidget(self.log_output)

    def toggle_sweep_ui(self):
        enabled = self.chk_sweep.isChecked()
        self.wl_stop.setEnabled(enabled); self.wl_steps.setEnabled(enabled)

    def refresh_range_fields(self):
        while self.range_form.count():
            child = self.range_form.takeAt(0)
            if child.widget(): child.widget().deleteLater()
        
        self.edit_lambda = QLineEdit("0.2e-6, 0.4e-6"); self.range_form.addRow("Lambda Range:", self.edit_lambda)
        self.edit_dc = QLineEdit("0.1, 0.9"); self.range_form.addRow("DC Range:", self.edit_dc)
        self.edit_w = QLineEdit("0.4e-6, 0.6e-6"); self.range_form.addRow("Width Range:", self.edit_w)
        self.edit_n = QLineEdit("2, 500"); self.range_form.addRow("N Periods Range:", self.edit_n)
        
        if self.rad_apodized.isChecked():
            self.edit_ds = QLineEdit("0.01e-6, 0.15e-6"); self.range_form.addRow("Max Delta_s:", self.edit_ds)
            self.edit_p = QLineEdit("1, 10"); self.range_form.addRow("Shape Factor (P):", self.edit_p)
            self.edit_m = QLineEdit("1, 5"); self.range_form.addRow("Decay Factor (M):", self.edit_m)

    def collect_config(self):
        def parse(txt): return tuple(map(float, txt.replace(" ", "").split(',')))
        cfg = copy.deepcopy(DEFAULT_CONFIG)
        cfg['ga_params'].update({
            "population_size": self.pop_size.value(), "mutation_rate": self.mut_rate.value(),
            "num_generations": self.num_gen.value(), "enable_convergence_check": self.conv_check.isChecked(),
            "convergence_patience_ratio": self.p_ratio.value(), "min_convergence_patience": self.min_p.value()
        })
        ranges = {"Lambda_range": parse(self.edit_lambda.text()), "DC_range": parse(self.edit_dc.text()),
                  "w_range": parse(self.edit_w.text()), "N_range": parse(self.edit_n.text()), "w_c_range_max_ratio": 0.8}
        
        if self.rad_apodized.isChecked():
            ranges.update({"mode": "apodized", "delta_s_max_range": parse(self.edit_ds.text()),
                           "P_range": parse(self.edit_p.text()), "M_range": parse(self.edit_m.text())})
        else: ranges["mode"] = "uniform"
        
        cfg["ga_ranges"] = ranges
        return cfg

    def start_opt(self, restart=True):
        start_idx = 0
        if not restart and os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, 'r') as f:
                    state = json.load(f); start_idx = state.get("next_index", 0)
                    self.log_output.append(f"--- Retomando do índice {start_idx} ---")
            except: pass
        try:
            config = self.collect_config()
            sweep = {'enabled': self.chk_sweep.isChecked(), 'start': self.wl_start.value(), 
                     'stop': self.wl_stop.value(), 'steps': self.wl_steps.value()}
            self.worker = OptimizationWorker(config, sweep, start_idx)
            self.worker.progress_sig.connect(self.log_output.append)
            self.worker.finished_sig.connect(self.handle_finish)
            self.btn_start.setEnabled(False); self.btn_resume.setEnabled(False); self.btn_stop.setEnabled(True)
            self.worker.start()
        except Exception as e: QMessageBox.critical(self, "Erro", str(e))

    def stop_opt(self):
        if self.worker: 
            self.worker.stop()
            self.log_output.append("\nSolicitando parada ao motor de otimização...")

    def handle_finish(self, result):
        self.btn_start.setEnabled(True); self.btn_resume.setEnabled(True); self.btn_stop.setEnabled(False)
        self.log_output.append(f"\nFinalizado. Status: {result}")

if __name__ == "__main__":
    app = QApplication(sys.argv); window = BraggSupervisorUI(); window.show(); sys.exit(app.exec_())