import csv
import json
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import cm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.metrics import confusion_matrix

from Projeto_Solar_RNA.dataset_solar import SolarDataset, CLASSES, get_class_group, GROUP_COLORS
from Projeto_Solar_RNA.neural_network import SingleLayerPerceptron, MLPSolarNetwork, evaluate_network
from Projeto_Solar_RNA.visualization_solar import SolarNetworkView, CLASSES_SHORT


class SolarApp:
    def __init__(self, root):
        self.root = root
        root.title('Laboratório RNA Solar | Inspeção Térmica e Diagnóstico de Painéis Fotovoltaicos')
        root.geometry('1400x900')
        root.minsize(1200, 800)
        
        # Load Dataset
        self.status = tk.StringVar(value='Carregando dataset infravermelho...')
        self.dataset = SolarDataset()
        self.train_idx, self.val_idx, self.test_idx = self.dataset.split_dataset(seed=42)
        
        # Preload initial subset for fluid GUI responsiveness
        self.dataset.preload_subset(self.train_idx, max_count=500)
        self.dataset.preload_subset(self.test_idx, max_count=200)

        # App Variables
        self.architecture_var = tk.StringVar(value='MLP (Camada Oculta 4x6 = 24 Detectores)')
        self.partition_var = tk.StringVar(value='Teste')
        self.class_filter_var = tk.StringVar(value='Todas')
        self.sample_idx_var = tk.IntVar(value=0)
        
        self.learning_rate_var = tk.StringVar(value='0.01')
        self.epochs_var = tk.StringVar(value='10')
        self.speed_var = tk.StringVar(value='Detalhado (1/quadro)')
        self.delay_var = tk.DoubleVar(value=50.0)
        
        self.intensity_var = tk.DoubleVar(value=1.0)
        self.erase_var = tk.BooleanVar(value=False)
        
        self.inspector_target_var = tk.StringVar(value='Classe de Saída')
        self.selected_output_var = tk.StringVar(value='0')
        self.selected_hidden_var = tk.StringVar(value='0')
        
        self.progress_var = tk.StringVar(value='Época 0 / 10 | 0 Atualizações')
        self.result_var = tk.StringVar(value='Diagnóstico: --')
        self.group_badge_var = tk.StringVar(value='Grupo: --')
        self.show_edges_var = tk.BooleanVar(value=True)
        
        # Current thermal image (40, 24)
        self.drawing = np.full((40, 24), 0.25, dtype=np.float32)
        self.current_feature_vector = self.drawing.flatten()
        self.current_label = 0
        self.is_manual_drawing = False

        # Neural Models
        self.perceptron = SingleLayerPerceptron(num_inputs=960, num_classes=12)
        self.mlp = MLPSolarNetwork(num_inputs=960, hidden_grid=(4, 6), num_classes=12)
        self.active_model = self.mlp

        # Training State
        self.epoch = 0
        self.position = 0
        self.updates_count = 0
        self.running = False
        self.pending_phase = 0  # 0: Forward, 1: Eval, 2: Update
        self.job = None
        self.order = None
        
        self.history = []

        # GUI Construction
        self._build_gui()
        self.load_current_sample()
        self.refresh_view()
        self.status.set('Pronto: Selecione uma amostra ou injete uma falha sintética para testar!')

    def _build_gui(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('.', font=('Bahnschrift', 9), background='#0f172a', foreground='#f8fafc')
        style.configure('TFrame', background='#0f172a')
        style.configure('TLabelframe', background='#0f172a', foreground='#38bdf8')
        style.configure('TLabelframe.Label', background='#0f172a', foreground='#38bdf8', font=('Bahnschrift', 10, 'bold'))
        style.configure('TButton', padding=(6, 4))
        style.configure('Title.TLabel', font=('Bahnschrift', 16, 'bold'), foreground='#f8fafc')
        style.configure('Result.TLabel', font=('Bahnschrift', 13, 'bold'), foreground='#38bdf8')
        style.configure('Badge.TLabel', font=('Bahnschrift', 11, 'bold'), foreground='#2ecc71')
        style.configure('TCheckbutton', background='#0f172a', foreground='#f8fafc')

        # Custom Entry & Combobox Dark High-Contrast Styling
        style.configure('TEntry', fieldbackground='#1e293b', foreground='#ffffff', insertcolor='#ffffff')
        style.configure('TCombobox', fieldbackground='#1e293b', background='#1e293b', foreground='#ffffff', arrowcolor='#38bdf8')
        style.map('TCombobox',
                  fieldbackground=[('readonly', '#1e293b'), ('focus', '#1e293b')],
                  foreground=[('readonly', '#ffffff'), ('focus', '#ffffff')],
                  selectbackground=[('readonly', '#0284c7')],
                  selectforeground=[('readonly', '#ffffff')])

        # Popup dropdown listbox option styling for TCombobox
        self.root.option_add('*TCombobox*Listbox.background', '#1e293b')
        self.root.option_add('*TCombobox*Listbox.foreground', '#ffffff')
        self.root.option_add('*TCombobox*Listbox.selectBackground', '#0284c7')
        self.root.option_add('*TCombobox*Listbox.selectForeground', '#ffffff')
        self.root.option_add('*TCombobox*Listbox.font', ('Bahnschrift', 9))

        # Notebook tabs styling
        style.configure('TNotebook', background='#0f172a', borderwidth=0)
        style.configure('TNotebook.Tab', background='#1e293b', foreground='#cbd5e1', padding=(10, 4))
        style.map('TNotebook.Tab',
                  background=[('selected', '#0284c7')],
                  foreground=[('selected', '#ffffff')])

        main_frame = ttk.Frame(self.root, padding=12)
        main_frame.pack(fill='both', expand=True)

        # --- HEADER ---
        header = ttk.Frame(main_frame)
        header.pack(fill='x', pady=(0, 8))
        ttk.Label(header, text='RNA SOLAR | Inspeção Térmica e Diagnóstico em Tempo Real (Matriz 24x40)',
                  style='Title.TLabel').pack(side='left')
        ttk.Label(header, textvariable=self.progress_var, font=('Consolas', 10, 'bold'),
                  foreground='#38bdf8').pack(side='right', padx=10)

        ttk.Separator(main_frame).pack(fill='x', pady=(0, 8))

        # --- THREE COLUMN DASHBOARD ---
        content = ttk.Frame(main_frame)
        content.pack(fill='both', expand=True)

        # COLUMN 1: Left IR Thermal Panel & Fault Simulator (Width 290)
        left_col = ttk.Frame(content, width=290)
        left_col.pack(side='left', fill='y', padx=(0, 10))
        self._build_left_panel(left_col)

        # COLUMN 2: Center Neural Engine Canvas (Width 570)
        center_col = ttk.Frame(content)
        center_col.pack(side='left', fill='both', expand=True, padx=(0, 10))
        self._build_center_panel(center_col)

        # COLUMN 3: Right Diagnostics & Metrics (Width 440)
        right_col = ttk.Frame(content, width=440)
        right_col.pack(side='right', fill='both', expand=False)
        self._build_right_panel(right_col)

    def _build_left_panel(self, parent):
        ttk.Label(parent, text='Painel Térmico IR (40 Linhas x 24 Colunas)', font=('Bahnschrift', 10, 'bold')).pack(anchor='w')

        # IR Image Canvas (24x40 scaled to 240x400)
        self.image_canvas = tk.Canvas(parent, width=240, height=400, bg='#020617', highlightthickness=1, highlightbackground='#334155')
        self.image_canvas.pack(anchor='w', pady=6)
        
        self.image_canvas.bind('<Button-1>', self.paint)
        self.image_canvas.bind('<B1-Motion>', self.paint)
        self.image_canvas.bind('<Button-3>', lambda e: self.paint(e, erase=True))

        # Paint Controls
        paint_frame = ttk.Frame(parent)
        paint_frame.pack(fill='x', pady=2)
        ttk.Checkbutton(paint_frame, text='Borracha', variable=self.erase_var).pack(side='left')
        ttk.Label(paint_frame, text='  Intensidade:').pack(side='left')
        ttk.Scale(paint_frame, from_=0.0, to=1.0, variable=self.intensity_var).pack(side='left', fill='x', expand=True)

        # Synthetic Fault Injection Tools
        fault_frame = ttk.LabelFrame(parent, text='Simulador de Falhas Térmicas de Campo', padding=6)
        fault_frame.pack(fill='x', pady=6)

        row_f1 = ttk.Frame(fault_frame)
        row_f1.pack(fill='x', pady=2)
        ttk.Button(row_f1, text='Ponto Quente (Hot-Spot)', command=lambda: self.inject_fault('Hot-Spot')).pack(side='left', padx=2)
        ttk.Button(row_f1, text='Diodo 1/3', command=lambda: self.inject_fault('Diode')).pack(side='left', padx=2)

        row_f2 = ttk.Frame(fault_frame)
        row_f2.pack(fill='x', pady=2)
        ttk.Button(row_f2, text='Módulo Desconectado', command=lambda: self.inject_fault('Offline-Module')).pack(side='left', padx=2)
        ttk.Button(row_f2, text='Vegetação/Sombra', command=lambda: self.inject_fault('Shadowing')).pack(side='left', padx=2)

        btn_act_frame = ttk.Frame(parent)
        btn_act_frame.pack(fill='x', pady=4)
        ttk.Button(btn_act_frame, text='Limpar Painel', command=self.clear_drawing).pack(side='left')
        ttk.Button(btn_act_frame, text='Classificar Agora', command=self.classify_current).pack(side='left', padx=4)

        ttk.Label(parent, textvariable=self.result_var, style='Result.TLabel').pack(anchor='w', pady=2)
        ttk.Label(parent, textvariable=self.group_badge_var, style='Badge.TLabel').pack(anchor='w', pady=2)
        
        ttk.Separator(parent).pack(fill='x', pady=6)

        # Dataset Browser
        ttk.Label(parent, text='Dataset Real (20.000 Imagens)', font=('Bahnschrift', 10, 'bold')).pack(anchor='w')

        f_part = ttk.Frame(parent)
        f_part.pack(fill='x', pady=2)
        ttk.Label(f_part, text='Divisão:').pack(side='left')
        cb_part = ttk.Combobox(f_part, textvariable=self.partition_var, values=['Treino', 'Validação', 'Teste'], state='readonly', width=12)
        cb_part.pack(side='right')
        cb_part.bind('<<ComboboxSelected>>', lambda e: self.on_partition_change())

        f_cls = ttk.Frame(parent)
        f_cls.pack(fill='x', pady=2)
        ttk.Label(f_cls, text='Classe:').pack(side='left')
        cb_cls = ttk.Combobox(f_cls, textvariable=self.class_filter_var, values=['Todas'] + CLASSES, state='readonly', width=12)
        cb_cls.pack(side='right')
        cb_cls.bind('<<ComboboxSelected>>', lambda e: self.on_partition_change())

        f_nav = ttk.Frame(parent)
        f_nav.pack(fill='x', pady=4)
        ttk.Button(f_nav, text='◄ Anterior', command=self.prev_sample).pack(side='left')
        ttk.Button(f_nav, text='Próximo ►', command=self.next_sample).pack(side='left', padx=4)
        
        self.sample_info_label = ttk.Label(parent, text='Amostra 0 / 0', font=('Consolas', 8))
        self.sample_info_label.pack(anchor='w', pady=2)

    def _build_center_panel(self, parent):
        # Top Controls: Architecture & Spatial Inspector Targets
        top_bar = ttk.Frame(parent)
        top_bar.pack(fill='x', pady=(0, 6))

        ttk.Label(top_bar, text='Modelo:').pack(side='left')
        cb_arch = ttk.Combobox(top_bar, textvariable=self.architecture_var,
                               values=['Perceptron Simples (1 Camada)', 'MLP (Camada Oculta 4x6 = 24 Detectores)'],
                               state='readonly', width=32)
        cb_arch.pack(side='left', padx=(4, 16))
        cb_arch.bind('<<ComboboxSelected>>', lambda e: self.on_architecture_change())

        ttk.Label(top_bar, text='Auditar Classe:').pack(side='left')
        cb_focus = ttk.Combobox(top_bar, textvariable=self.selected_output_var,
                                values=[str(i) for i in range(12)], state='readonly', width=5)
        cb_focus.pack(side='left', padx=4)
        cb_focus.bind('<<ComboboxSelected>>', lambda e: self.refresh_view())

        # Button to Enable/Disable Connection Lines (Edges)
        btn_edges = ttk.Checkbutton(top_bar, text='Exibir Linhas (Conexões)',
                                    variable=self.show_edges_var, command=self.refresh_view)
        btn_edges.pack(side='right', padx=6)

        # Neural Network Canvas View
        self.network_view = SolarNetworkView(parent, height=450)
        self.network_view.pack(fill='both', expand=True, pady=4)

        # Interactive Training Controls
        controls_frame = ttk.LabelFrame(parent, text='Controles de Treinamento e Animação', padding=8)
        controls_frame.pack(fill='x', pady=6)

        row1 = ttk.Frame(controls_frame)
        row1.pack(fill='x', pady=2)

        self.btn_step = ttk.Button(row1, text='Passo (Step)', command=self.step_one_phase)
        self.btn_step.pack(side='left', padx=2)

        self.btn_train = ttk.Button(row1, text='Treinar (Play)', command=self.toggle_training)
        self.btn_train.pack(side='left', padx=2)

        ttk.Button(row1, text='Reiniciar Pesos', command=self.reset_network).pack(side='left', padx=2)

        ttk.Label(row1, text='  Velocidade:').pack(side='left')
        cb_speed = ttk.Combobox(row1, textvariable=self.speed_var,
                                values=['Detalhado (1/quadro)', 'Rápido (10/quadro)', 'Ultra (50/quadro)'],
                                state='readonly', width=18)
        cb_speed.pack(side='left', padx=4)

        row2 = ttk.Frame(controls_frame)
        row2.pack(fill='x', pady=4)

        ttk.Label(row2, text='Taxa (η):').pack(side='left')
        ttk.Entry(row2, textvariable=self.learning_rate_var, width=6).pack(side='left', padx=(2, 10))

        ttk.Label(row2, text='Épocas:').pack(side='left')
        ttk.Entry(row2, textvariable=self.epochs_var, width=5).pack(side='left', padx=(2, 10))

        ttk.Label(row2, text='Delay (ms):').pack(side='left')
        ttk.Scale(row2, from_=1.0, to=200.0, variable=self.delay_var).pack(side='left', fill='x', expand=True)

    def _build_right_panel(self, parent):
        ttk.Label(parent, text='Métricas e Auditoria dos Neurônios', font=('Bahnschrift', 10, 'bold')).pack(anchor='w', pady=(0, 4))

        notebook = ttk.Notebook(parent)
        notebook.pack(fill='both', expand=True)

        # Tab 1: Curves
        tab_curves = ttk.Frame(notebook)
        notebook.add(tab_curves, text='Curvas de Treino')
        self.fig_curves = Figure(figsize=(4.2, 4.0), dpi=90, facecolor='#0f172a')
        self.ax_loss = self.fig_curves.add_subplot(211)
        self.ax_acc = self.fig_curves.add_subplot(212)
        self.fig_curves.tight_layout(pad=2.0)
        self.canvas_curves = FigureCanvasTkAgg(self.fig_curves, master=tab_curves)
        self.canvas_curves.get_tk_widget().pack(fill='both', expand=True)

        # Tab 2: Confusion Matrix
        tab_cm = ttk.Frame(notebook)
        notebook.add(tab_cm, text='Matriz de Confusão')
        self.fig_cm = Figure(figsize=(4.2, 4.0), dpi=90, facecolor='#0f172a')
        self.ax_cm = self.fig_cm.add_subplot(111)
        self.fig_cm.tight_layout(pad=2.0)
        self.canvas_cm = FigureCanvasTkAgg(self.fig_cm, master=tab_cm)
        self.canvas_cm.get_tk_widget().pack(fill='both', expand=True)

        # Tab 3: Weight Audit Heatmap (Receptive Field 24x40)
        tab_audit = ttk.Frame(notebook)
        notebook.add(tab_audit, text='Campo Receptivo 24x40')
        self.fig_audit = Figure(figsize=(4.2, 4.0), dpi=90, facecolor='#0f172a')
        self.ax_audit = self.fig_audit.add_subplot(111)
        self.fig_audit.tight_layout(pad=2.0)
        self.canvas_audit = FigureCanvasTkAgg(self.fig_audit, master=tab_audit)
        self.canvas_audit.get_tk_widget().pack(fill='both', expand=True)

        # Export actions
        export_frame = ttk.Frame(parent)
        export_frame.pack(fill='x', pady=6)
        ttk.Button(export_frame, text='Exportar Pesos (CSV)', command=self.export_csv).pack(side='left', padx=4)
        ttk.Button(export_frame, text='Exportar Histórico (JSON)', command=self.export_json).pack(side='left', padx=4)

        self.update_plots()

    # --- SIMULATION & FAULT INJECTION ---
    def inject_fault(self, anomaly_type):
        synth_img = SolarDataset.generate_synthetic_anomaly(anomaly_type=anomaly_type)
        self.drawing = synth_img.copy()
        self.current_feature_vector = synth_img.flatten()
        self.is_manual_drawing = True
        self.render_image_canvas(self.drawing)
        self.classify_current()

    def paint(self, event, erase=False):
        col = int(event.x // 6)   # 240 / 24 = 10 -> col width
        row = int(event.y // 10)  # 400 / 40 = 10 -> row height
        
        col = max(0, min(23, col))
        row = max(0, min(39, row))

        val = 0.25 if (erase or self.erase_var.get()) else self.intensity_var.get()
        
        for r in range(max(0, row - 1), min(40, row + 2)):
            for c in range(max(0, col - 1), min(24, col + 2)):
                self.drawing[r, c] = val

        self.current_feature_vector = self.drawing.flatten()
        self.is_manual_drawing = True
        self.render_image_canvas(self.drawing)
        self.refresh_view()

    def clear_drawing(self):
        self.drawing.fill(0.25)  # Nominal baseline
        self.current_feature_vector = self.drawing.flatten()
        self.is_manual_drawing = True
        self.render_image_canvas(self.drawing)
        self.refresh_view()

    def render_image_canvas(self, img_2d):
        arr_uint8 = (np.clip(img_2d, 0, 1) * 255).astype(np.uint8)
        colored = Image.fromarray(cm.inferno(arr_uint8 / 255.0, bytes=True))
        resized = colored.resize((240, 400), Image.Resampling.NEAREST)
        
        self.tk_image = ImageTk.PhotoImage(resized)
        self.image_canvas.create_image(0, 0, anchor='nw', image=self.tk_image)

    # --- DATASET NAVIGATION ---
    def get_current_partition_indices(self):
        part = self.partition_var.get()
        cls_filter = self.class_filter_var.get()
        
        if part == 'Treino':
            indices = self.train_idx
        elif part == 'Validação':
            indices = self.val_idx
        else:
            indices = self.test_idx

        if cls_filter != 'Todas':
            target_cls_idx = CLASSES.index(cls_filter)
            indices = [i for i in indices if self.dataset.labels[i] == target_cls_idx]

        return indices

    def on_partition_change(self):
        self.sample_idx_var.set(0)
        self.load_current_sample()

    def load_current_sample(self):
        indices = self.get_current_partition_indices()
        if len(indices) == 0:
            self.sample_info_label.config(text="Nenhuma amostra encontrada")
            return
            
        curr_i = max(0, min(len(indices) - 1, self.sample_idx_var.get()))
        self.sample_idx_var.set(curr_i)
        
        real_idx = indices[curr_i]
        img_2d = self.dataset.load_image(real_idx)
        self.drawing = img_2d.copy()
        self.current_feature_vector = img_2d.flatten()
        self.current_label = self.dataset.labels[real_idx]
        self.is_manual_drawing = False

        self.render_image_canvas(self.drawing)
        self.sample_info_label.config(text=f"Amostra {curr_i + 1} / {len(indices)} | Rótulo: {CLASSES[self.current_label]}")
        self.refresh_view()

    def prev_sample(self):
        self.sample_idx_var.set(self.sample_idx_var.get() - 1)
        self.load_current_sample()

    def next_sample(self):
        self.sample_idx_var.set(self.sample_idx_var.get() + 1)
        self.load_current_sample()

    # --- ARCHITECTURE & MODEL SWITCHING ---
    def on_architecture_change(self):
        arch = self.architecture_var.get()
        if 'MLP' in arch:
            self.mlp = MLPSolarNetwork(num_inputs=960, hidden_grid=(4, 6), num_classes=12)
            self.active_model = self.mlp
        else:
            self.active_model = self.perceptron
        self.reset_network()

    def reset_network(self):
        self.stop_training()
        arch = self.architecture_var.get()
        if 'MLP' in arch:
            self.mlp = MLPSolarNetwork(num_inputs=960, hidden_grid=(4, 6), num_classes=12)
            self.active_model = self.mlp
        else:
            self.perceptron.reset_weights()
            self.active_model = self.perceptron

        self.epoch = 0
        self.position = 0
        self.updates_count = 0
        self.history.clear()
        self.progress_var.set("Época 0 / 10 | 0 Atualizações")
        self.result_var.set("Diagnóstico: --")
        self.group_badge_var.set("Grupo: --")
        self.refresh_view()
        self.update_plots()

    # --- INFERENCE & CLASSIFICATION ---
    def classify_current(self):
        pred, scores, probs = self.active_model.forward(self.current_feature_vector)
        cls_name = CLASSES[pred]
        prob_pct = probs[pred] * 100.0
        grp_name = get_class_group(cls_name)
        
        self.result_var.set(f"Pred: {cls_name} ({prob_pct:.1f}%)")
        self.group_badge_var.set(f"Grupo: {grp_name}")
        self.refresh_view()

    def refresh_view(self):
        pred, scores, probs = self.active_model.forward(self.current_feature_vector)
        sel_output = int(self.selected_output_var.get())
        
        is_mlp = isinstance(self.active_model, MLPSolarNetwork)
        hidden_acts = self.active_model.get_hidden_activations() if is_mlp else None
        
        if is_mlp:
            w_matrix, b_val = self.active_model.W2, self.active_model.b2
        else:
            w_matrix, b_val = self.active_model.weights, self.active_model.biases

        target_lbl = None if self.is_manual_drawing else self.current_label

        self.network_view.update_view(
            features=self.current_feature_vector,
            weights=w_matrix,
            biases=b_val,
            scores=scores,
            probs=probs,
            selected_output=sel_output,
            winner=pred,
            target=target_lbl,
            is_mlp=is_mlp,
            hidden_activations=hidden_acts,
            hidden_grid_shape=(4, 6),
            show_edges=self.show_edges_var.get()
        )
        self.update_audit_plot()

    # --- TRAINING LOOP ---
    def toggle_training(self):
        if self.running:
            self.stop_training()
        else:
            self.start_training()

    def start_training(self):
        self.running = True
        self.btn_train.config(text="Pausar")
        self.order = list(self.train_idx)
        np.random.default_rng().shuffle(self.order)
        self.run_training_loop()

    def stop_training(self):
        self.running = False
        self.btn_train.config(text="Treinar (Play)")
        if self.job:
            self.root.after_cancel(self.job)
            self.job = None

    def step_one_phase(self):
        if self.order is None or self.position >= len(self.order):
            self.order = list(self.train_idx)
            np.random.default_rng().shuffle(self.order)
            self.position = 0

        real_idx = self.order[self.position]
        x = self.dataset.get_feature_vector(real_idx)
        target = self.dataset.labels[real_idx]
        lr = float(self.learning_rate_var.get())

        if self.pending_phase == 0:
            self.active_model.train_step_phase1_forward(x, target)
            self.drawing = x.reshape(40, 24)
            self.current_feature_vector = x
            self.current_label = target
            self.render_image_canvas(self.drawing)
            self.pending_phase = 1
        elif self.pending_phase == 1:
            had_error, loss = self.active_model.train_step_phase2_eval()
            self.pending_phase = 2
        elif self.pending_phase == 2:
            self.active_model.train_step_phase3_update(lr)
            if self.active_model.last_update_had_error:
                self.updates_count += 1
            self.position += 1
            self.pending_phase = 0

        max_epochs = int(self.epochs_var.get())
        self.progress_var.set(f"Época {self.epoch + 1} / {max_epochs} | {self.updates_count} Atualizações")
        self.refresh_view()

    def run_training_loop(self):
        if not self.running:
            return

        max_epochs = int(self.epochs_var.get())
        lr = float(self.learning_rate_var.get())
        speed = self.speed_var.get()
        
        steps_per_frame = 1
        if '10' in speed:
            steps_per_frame = 10
        elif '50' in speed:
            steps_per_frame = 50

        for _ in range(steps_per_frame):
            if self.position >= len(self.train_idx):
                self.position = 0
                self.epoch += 1
                np.random.default_rng().shuffle(self.order)

                if self.epoch >= max_epochs:
                    self.stop_training()
                    messagebox.showinfo("Treinamento Concluído", f"Treinamento de {max_epochs} épocas finalizado com sucesso!")
                    return

            real_idx = self.order[self.position]
            x = self.dataset.get_feature_vector(real_idx)
            target = self.dataset.labels[real_idx]

            pred, had_error, loss = self.active_model.train_step(x, target, lr=lr)
            if had_error:
                self.updates_count += 1

            self.position += 1

        self.current_feature_vector = x
        self.current_label = target
        self.drawing = x.reshape(40, 24)
        self.render_image_canvas(self.drawing)

        # Record live metric checkpoint
        step_progress = self.epoch + (self.position / max(len(self.train_idx), 1))
        tr_acc, tr_loss, _, _ = evaluate_network(self.active_model, self.dataset, self.train_idx[:200])
        val_acc, val_loss, _, _ = evaluate_network(self.active_model, self.dataset, self.val_idx[:150])

        self.history.append({
            'epoch': round(step_progress, 2),
            'train_acc': tr_acc,
            'val_acc': val_acc,
            'loss': tr_loss
        })

        self.progress_var.set(f"Época {self.epoch + 1} / {max_epochs} | {self.updates_count} Atualizações")
        self.refresh_view()
        self.update_plots()

        delay_ms = int(self.delay_var.get())
        self.job = self.root.after(delay_ms, self.run_training_loop)

    # --- METRICS PLOTS & AUDIT ---
    def update_plots(self):
        self.ax_loss.clear()
        self.ax_acc.clear()

        # Dark Theme Styling for Matplotlib
        for ax in [self.ax_loss, self.ax_acc]:
            ax.set_facecolor('#020617')
            ax.tick_params(colors='#cbd5e1', labelsize=7)
            for spine in ax.spines.values():
                spine.set_color('#334155')

        epochs_list = [h['epoch'] for h in self.history]
        loss_list = [h['loss'] for h in self.history]
        tr_acc_list = [h['train_acc'] * 100 for h in self.history]
        val_acc_list = [h['val_acc'] * 100 for h in self.history]

        self.ax_loss.plot(epochs_list, loss_list, color='#ef4444', marker='o', label='Loss (Treino)')
        self.ax_loss.set_title('Perda (Cross-Entropy)', color='#f8fafc', fontsize=9)
        self.ax_loss.grid(True, linestyle='--', color='#334155', alpha=0.5)

        self.ax_acc.plot(epochs_list, tr_acc_list, color='#22c55e', marker='o', label='Treino (%)')
        self.ax_acc.plot(epochs_list, val_acc_list, color='#38bdf8', marker='s', label='Validação (%)')
        self.ax_acc.set_title('Acurácia (%)', color='#f8fafc', fontsize=9)
        self.ax_acc.set_xlabel('Época', color='#cbd5e1', fontsize=8)
        self.ax_acc.legend(fontsize=7, facecolor='#0f172a', edgecolor='#334155', labelcolor='#f8fafc')
        self.ax_acc.grid(True, linestyle='--', color='#334155', alpha=0.5)

        self.fig_curves.tight_layout(pad=1.5)
        self.canvas_curves.draw()

        # Update Confusion Matrix
        self.ax_cm.clear()
        self.ax_cm.set_facecolor('#020617')
        self.ax_cm.tick_params(colors='#cbd5e1', labelsize=7)
        for spine in self.ax_cm.spines.values():
            spine.set_color('#334155')

        val_sub = self.val_idx[:300]
        acc, avg_l, preds, targets = evaluate_network(self.active_model, self.dataset, val_sub)
        cm_matrix = confusion_matrix(targets, preds, labels=list(range(12)))

        cax = self.ax_cm.matshow(cm_matrix, cmap='Blues')
        self.ax_cm.set_title(f"Matriz de Confusão Val (Acc: {acc*100:.1f}%)", color='#f8fafc', fontsize=9)
        self.ax_cm.set_xticks(range(12))
        self.ax_cm.set_yticks(range(12))
        self.ax_cm.set_xticklabels([str(i) for i in range(12)], color='#cbd5e1', fontsize=7)
        self.ax_cm.set_yticklabels([str(i) for i in range(12)], color='#cbd5e1', fontsize=7)
        self.fig_cm.tight_layout(pad=1.5)
        self.canvas_cm.draw()

    def update_audit_plot(self):
        self.ax_audit.clear()
        self.ax_audit.set_facecolor('#020617')
        
        sel_output = int(self.selected_output_var.get())
        rf = self.active_model.get_receptive_field(sel_output, shape=(40, 24))

        vmax = max(float(np.max(np.abs(rf))), 0.001)
        im = self.ax_audit.imshow(rf, cmap='bwr', vmin=-vmax, vmax=vmax, aspect='auto')
        self.ax_audit.set_title(f"Campo Receptivo 24x40 ({CLASSES[sel_output]})", color='#f8fafc', fontsize=9)
        self.ax_audit.set_xticks([])
        self.ax_audit.set_yticks([])
        self.fig_audit.tight_layout(pad=1.5)
        self.canvas_audit.draw()

    # --- EXPORT AUDIT DATA ---
    def export_csv(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return

        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Class_Index', 'Class_Name', 'Pixel_Row', 'Pixel_Col', 'Weight_Value'])
            
            for cls_idx in range(12):
                rf = self.active_model.get_receptive_field(cls_idx, shape=(40, 24))
                for r in range(40):
                    for c in range(24):
                        writer.writerow([cls_idx, CLASSES[cls_idx], r, c, float(rf[r, c])])

        messagebox.showinfo("Exportação Concluída", f"Pesos 24x40 exportados com sucesso em:\n{file_path}")

    def export_json(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if not file_path:
            return

        data = {
            'architecture': self.architecture_var.get(),
            'epoch': self.epoch,
            'updates_count': self.updates_count,
            'history': self.history
        }

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

        messagebox.showinfo("Exportação Concluída", f"Histórico exportado com sucesso em:\n{file_path}")


def main():
    root = tk.Tk()
    app = SolarApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
