import csv
import json
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.metrics import confusion_matrix

from dataset import load_reference_dataset
from perceptron import MultiClassPerceptron
from visualization import NetworkView


class PerceptronApp:
    def __init__(self, root):
        self.root = root
        root.title('Laboratorio Perceptron | Digitos 0-9')
        root.geometry('1280x850')
        root.minsize(1100, 760)
        self.data = load_reference_dataset()
        self.drawing = np.zeros(64)
        self.features = self.drawing.copy()
        self.drawing_source = self.source = 'Desenho manual'
        self.pending = self.last_step = self.job = self.order = None
        self.running = self.classifying = False
        self.reference_cursor = self.epoch = self.position = self.mistakes = 0
        self.history, self.records = [], []
        self.rate = tk.StringVar(value='0.1')
        self.epochs = tk.StringVar(value='20')
        self.delay = tk.DoubleVar(value=100)
        self.intensity = tk.DoubleVar(value=1.0)
        self.erase = tk.BooleanVar(value=False)
        self.selected = tk.StringVar(value='0')
        self.digit = tk.StringVar(value='0')
        self.partition = tk.StringVar(value='Teste')
        self.speed = tk.StringVar(value='Detalhado')
        self.status = tk.StringVar(value='Rede inicializada / pesos zerados')
        self.result = tk.StringVar(value='Resultado: --')
        self.progress = tk.StringVar(value='Epoca 0 | 0 atualizacoes')
        self.model = MultiClassPerceptron()
        self.snapshot_weights = self.model.weights.copy()
        self.snapshot_biases = self.model.biases.copy()
        self.snapshot_updates = 0
        self.snapshot_label = tk.StringVar(value='Referencia: pesos iniciais')
        self.score_label = tk.StringVar(value='')
        self.random = np.random.default_rng(42)
        self._build()
        self.refresh()
        root.protocol('WM_DELETE_WINDOW', self.close)

    def _build(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('.', font=('Bahnschrift', 10), background='#f7faf9', foreground='#24332f')
        style.configure('TButton', padding=(8, 6))
        style.configure('Title.TLabel', font=('Bahnschrift', 20, 'bold'))
        style.configure('Result.TLabel', font=('Bahnschrift', 17, 'bold'), foreground='#087f70')
        frame = ttk.Frame(self.root, padding=16)
        frame.pack(fill='both', expand=True)
        ttk.Label(frame, text='PERCEPTRON / laboratorio de digitos', style='Title.TLabel').pack(anchor='w')
        ttk.Label(frame, textvariable=self.progress).pack(anchor='w', pady=(4, 12))
        body = ttk.Frame(frame)
        body.pack(fill='both', expand=True)
        self.sidebar = ttk.Frame(body, width=258)
        self.sidebar.pack(side='left', fill='y', padx=(0, 18))
        ttk.Label(self.sidebar, text='Entrada / 8 x 8 pixels').pack(anchor='w')
        self.pixel_canvas = tk.Canvas(self.sidebar, width=240, height=240, highlightthickness=0)
        self.pixel_canvas.pack(anchor='w', pady=4)
        self.cells = []
        for index in range(64):
            column, row = index % 8, index // 8
            self.cells.append(self.pixel_canvas.create_rectangle(
                column * 30, row * 30, (column + 1) * 30, (row + 1) * 30,
                fill='#000000', outline='#47564f'))
        self.pixel_canvas.bind('<Button-1>', self.paint)
        self.pixel_canvas.bind('<B1-Motion>', self.paint)
        self.pixel_canvas.bind('<Button-3>', lambda event: self.paint(event, erase=True))
        self.pixel_canvas.bind('<B3-Motion>', lambda event: self.paint(event, erase=True))
        ttk.Checkbutton(self.sidebar, text='Borracha', variable=self.erase).pack(anchor='w')
        ttk.Label(self.sidebar, text='Intensidade do pixel').pack(anchor='w')
        ttk.Scale(self.sidebar, from_=0, to=1, variable=self.intensity).pack(fill='x')
        actions = ttk.Frame(self.sidebar)
        actions.pack(fill='x', pady=4)
        ttk.Button(actions, text='Limpar', command=self.clear).pack(side='left')
        ttk.Button(actions, text='Classificar', command=self.classify).pack(side='left', padx=6)
        ttk.Label(self.sidebar, textvariable=self.result, style='Result.TLabel').pack(anchor='w', pady=4)
        ttk.Separator(self.sidebar).pack(fill='x', pady=5)
        ttk.Label(self.sidebar, text='Amostra de referencia').pack(anchor='w')
        reference = ttk.Frame(self.sidebar)
        reference.pack(fill='x', pady=4)
        ttk.Combobox(reference, values=list(range(10)), textvariable=self.digit,
                     state='readonly', width=3).pack(side='left')
        ttk.Combobox(reference, values=['Treino', 'Teste'], textvariable=self.partition,
                     state='readonly', width=8).pack(side='left', padx=5)
        ttk.Button(self.sidebar, text='Carregar proxima amostra', command=self.load_sample).pack(fill='x')
        ttk.Separator(self.sidebar).pack(fill='x', pady=6)
        ttk.Label(self.sidebar, text='Neuronio em foco').pack(anchor='w')
        focus = ttk.Combobox(self.sidebar, values=list(range(10)), textvariable=self.selected,
                             state='readonly', width=6)
        focus.pack(anchor='w', pady=4)
        focus.bind('<<ComboboxSelected>>', lambda event: self.refresh())
        self.tabs = ttk.Notebook(body)
        self.tabs.pack(side='left', fill='both', expand=True)
        network_tab = ttk.Frame(self.tabs, padding=10)
        self.tabs.add(network_tab, text='Rede ao vivo')
        self.tabs.bind('<<NotebookTabChanged>>', lambda event: self.refresh())
        controls = ttk.Frame(network_tab)
        controls.pack(fill='x')
        ttk.Label(controls, text='Taxa').pack(side='left')
        self.rate_entry = ttk.Entry(controls, textvariable=self.rate, width=6)
        self.rate_entry.pack(side='left', padx=(4, 12))
        ttk.Label(controls, text='Epocas').pack(side='left')
        self.epoch_entry = ttk.Spinbox(controls, from_=1, to=200, textvariable=self.epochs, width=5)
        self.epoch_entry.pack(side='left', padx=4)
        ttk.Combobox(controls, values=['Detalhado', 'Rapido (50 amostras/quadro)'],
                     textvariable=self.speed, state='readonly', width=27).pack(side='left', padx=6)
        commands = ttk.Frame(network_tab)
        commands.pack(fill='x', pady=8)
        self.run_button = ttk.Button(commands, text='Treinar', command=self.toggle_training)
        self.run_button.pack(side='left')
        ttk.Button(commands, text='Passo', command=self.single_step).pack(side='left', padx=6)
        ttk.Button(commands, text='Reiniciar rede', command=self.reset).pack(side='left')
        ttk.Label(commands, text='Intervalo (ms)').pack(side='left', padx=(12, 4))
        ttk.Scale(commands, from_=20, to=800, variable=self.delay, length=100).pack(side='left')
        ttk.Label(network_tab, textvariable=self.status, wraplength=700).pack(anchor='w', pady=6)
        self.network = NetworkView(network_tab)
        self.network.pack(fill='both', expand=True)
        self._build_audit()

    def _build_audit(self):
        ttk.Button(self.sidebar, text='Capturar pesos de referencia', command=self.capture).pack(fill='x')
        ttk.Button(self.sidebar, text='Exportar auditoria', command=self.export_dialog).pack(fill='x', pady=6)
        self.audit_tab = ttk.Frame(self.tabs, padding=10)
        self.metrics_tab = ttk.Frame(self.tabs, padding=10)
        data_tab = ttk.Frame(self.tabs, padding=10)
        self.tabs.add(self.audit_tab, text='Auditoria dos pesos')
        self.tabs.add(self.metrics_tab, text='Treinamento')
        self.tabs.add(data_tab, text='Base de referencia')
        ttk.Label(self.audit_tab, textvariable=self.snapshot_label).pack(anchor='w')
        ttk.Label(self.audit_tab, textvariable=self.score_label, wraplength=700).pack(anchor='w', pady=5)
        self.audit_figure = Figure(figsize=(8, 2.3), dpi=100, layout='constrained')
        self.audit_axes = self.audit_figure.subplots(1, 3)
        self.audit_canvas = FigureCanvasTkAgg(self.audit_figure, master=self.audit_tab)
        self.audit_canvas.get_tk_widget().pack(fill='x')
        table_frame = ttk.Frame(self.audit_tab)
        table_frame.pack(fill='both', expand=True, pady=8)
        columns = ('pixel', 'entrada', 'referencia', 'atual', 'delta', 'antes', 'passo', 'contribuicao')
        self.table = ttk.Treeview(table_frame, columns=columns, show='headings', height=10)
        headings = ('Pixel', 'Entrada', 'Referencia', 'Peso atual', 'Delta ref.', 'Antes passo', 'Delta passo', 'x * peso')
        for column, heading in zip(columns, headings):
            self.table.heading(column, text=heading)
            self.table.column(column, width=84, minwidth=60, anchor='e')
        scrollbar = ttk.Scrollbar(table_frame, orient='vertical', command=self.table.yview)
        self.table.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side='right', fill='y')
        self.table.pack(fill='both', expand=True)
        self.metrics_figure = Figure(figsize=(8, 6), dpi=100, layout='constrained')
        layout = self.metrics_figure.add_gridspec(2, 2)
        self.accuracy_axis = self.metrics_figure.add_subplot(layout[0, 0])
        self.error_axis = self.metrics_figure.add_subplot(layout[0, 1])
        self.confusion_axis = self.metrics_figure.add_subplot(layout[1, :])
        self.metrics_canvas = FigureCanvasTkAgg(self.metrics_figure, master=self.metrics_tab)
        self.metrics_canvas.get_tk_widget().pack(fill='both', expand=True)
        ttk.Label(data_tab, text='UCI / Digitos manuscritos',
                  font=('Bahnschrift', 15, 'bold')).pack(anchor='w', pady=8)
        ttk.Label(data_tab, text='1.797 imagens / 64 pixels / classes 0 a 9\n'
                  'Treino: 1.347 | Teste: 450 | Separacao estratificada | Semente: 42\n'
                  'Intensidades originais: 0 a 16 | Entrada da rede: 0 a 1',
                  justify='left').pack(anchor='w', pady=8)
        gallery = Figure(figsize=(8, 4), dpi=100, layout='constrained')
        for digit, axis in enumerate(gallery.subplots(2, 5).flat):
            sample = self.data.train_images[self.data.train_labels == digit][0]
            axis.imshow(sample.reshape(8, 8), cmap='gray', vmin=0, vmax=1)
            axis.set_title(str(digit))
            axis.set_axis_off()
        self.gallery_canvas = FigureCanvasTkAgg(gallery, master=data_tab)
        self.gallery_canvas.get_tk_widget().pack(fill='both', expand=True)
        ttk.Label(data_tab, text='Fonte: sklearn.datasets.load_digits / UCI ML Repository',
                  wraplength=700).pack(anchor='w', pady=10)

    def paint(self, event, erase=False):
        if self.running or self.classifying or self.pending is not None:
            return
        if 0 <= event.x < 240 and 0 <= event.y < 240:
            index = (event.y // 30) * 8 + event.x // 30
            self.drawing[index] = 0 if erase or self.erase.get() else self.intensity.get()
            self.drawing_source = 'Desenho manual'
            self.result.set('Resultado: --')
            self.draw_pixels()

    def draw_pixels(self):
        for index, cell in enumerate(self.cells):
            shade = int(self.drawing[index] * 255)
            self.pixel_canvas.itemconfigure(cell, fill=f'#{shade:02x}{shade:02x}{shade:02x}')

    def clear(self):
        if self.running or self.classifying or self.pending is not None:
            return
        self.drawing.fill(0)
        self.features = self.drawing.copy()
        self.drawing_source = self.source = 'Desenho manual'
        self.result.set('Resultado: --')
        self.draw_pixels()
        self.refresh()

    def load_sample(self):
        if self.running or self.classifying or self.pending is not None:
            return
        is_train = self.partition.get() == 'Treino'
        images = self.data.train_images if is_train else self.data.test_images
        labels = self.data.train_labels if is_train else self.data.test_labels
        indices = self.data.train_indices if is_train else self.data.test_indices
        candidates = np.flatnonzero(labels == int(self.digit.get()))
        index = candidates[self.reference_cursor % len(candidates)]
        self.reference_cursor += 1
        self.drawing = images[index].copy()
        self.features = self.drawing.copy()
        self.drawing_source = self.source = f'{self.partition.get()} / indice UCI {indices[index]} / rotulo {labels[index]}'
        self.status.set(self.source)
        self.result.set('Resultado: --')
        self.draw_pixels()
        self.refresh()

    def configure_training(self):
        try:
            rate, epochs = float(self.rate.get()), int(self.epochs.get())
            if not np.isfinite(rate) or rate <= 0 or not 1 <= epochs <= 200:
                raise ValueError
        except ValueError:
            messagebox.showerror('Parametros invalidos', 'Taxa positiva; epocas entre 1 e 200.')
            return False
        self.model.learning_rate = rate
        self.target_epochs = epochs
        self.rate_entry.configure(state='disabled')
        self.epoch_entry.configure(state='disabled')
        return True

    def toggle_training(self):
        if self.running:
            self.pause()
            return
        if self.classifying or not self.configure_training():
            return
        if self.epoch >= self.target_epochs:
            self.status.set('Treino concluido. Reinicie a rede para uma nova execucao.')
            return
        self.running = True
        self.run_button.configure(text='Pausar')
        self.tick()

    def pause(self):
        self.running = False
        self.run_button.configure(text='Continuar')
        if self.job is not None:
            self.root.after_cancel(self.job)
            self.job = None

    def single_step(self):
        if self.classifying:
            return
        self.pause()
        if self.configure_training():
            self.advance()

    def tick(self):
        self.job = None
        if not self.running:
            return
        if self.speed.get().startswith('Rapido'):
            for unused in range(50):
                if self.epoch >= self.target_epochs:
                    break
                if self.pending is None:
                    self.advance(render=False)
                while self.pending is not None:
                    self.advance(render=False)
            self.refresh()
        else:
            self.advance()
        if self.epoch >= self.target_epochs:
            self.pause()
            self.status.set(f'Treino concluido / {self.epoch} epocas / teste {self.history[-1]["test"]:.1%}')
        elif self.running:
            self.job = self.root.after(int(self.delay.get()), self.tick)

    def advance(self, render=True):
        if self.epoch >= self.target_epochs:
            return
        if self.pending is None:
            if self.order is None:
                self.order = self.random.permutation(len(self.data.train_labels))
            index = int(self.order[self.position])
            self.features = self.data.train_images[index].copy()
            label = int(self.data.train_labels[index])
            self.pending = {'index': index, 'label': label, 'phase': 'entrada'}
            self.source = f'Treino / indice UCI {self.data.train_indices[index]} / rotulo {label}'
            self.status.set(f'{self.source} / fase: entrada')
        elif self.pending['phase'] == 'entrada':
            self.pending['phase'] = 'somas'
            predicted = self.model.predict_one(self.features)
            self.status.set(f'{self.source} / antes da correcao: {predicted}')
        else:
            label, index = self.pending['label'], self.pending['index']
            step = self.model.train_one(self.features, label)
            self.last_step = step
            self.mistakes += int(step.updated)
            self.records.append({'epoch': self.epoch + 1, 'position': self.position + 1,
                                 'dataset_index': int(self.data.train_indices[index]),
                                 'label': label, 'predicted_before': step.predicted,
                                 'scores_before': step.scores_before.tolist(),
                                 'updated': step.updated, 'learning_rate': self.model.learning_rate})
            self.pending = None
            self.position += 1
            self.status.set(f'{self.source} / previsto antes: {step.predicted} / '
                            f'{"pesos corrigidos" if step.updated else "sem correcao"}')
            if self.position == len(self.data.train_labels):
                self.finish_epoch()
        self.progress.set(f'Epoca {self.epoch + (self.position > 0 or self.pending is not None)} / '
                          f'amostra {self.position} de {len(self.data.train_labels)} / '
                          f'{self.model.updates} atualizacoes')
        if render:
            self.refresh()

    def finish_epoch(self):
        self.epoch += 1
        train = self.model.evaluate(list(zip(self.data.train_images, self.data.train_labels)))
        test = self.model.evaluate(list(zip(self.data.test_images, self.data.test_labels)))
        self.history.append({'epoch': self.epoch, 'mistakes': self.mistakes, 'train': train, 'test': test})
        self.position = self.mistakes = 0
        self.order = None

    def classify(self):
        if self.classifying:
            return
        self.pause()
        if self.pending is not None:
            self.status.set('Conclua a amostra atual com Passo antes de classificar.')
            return
        if not np.any(self.drawing):
            self.result.set('Entrada vazia')
            return
        self.features = self.drawing.copy()
        self.source = self.drawing_source
        self.classifying = True
        self.classification_active = 0
        self.result.set('Calculando...')
        self.tabs.select(0)
        self.classification_tick()

    def classification_tick(self):
        self.job = None
        audit = self.model.inspect(self.features)
        active = self.classification_active
        self.network.show(self.features, self.model.weights, self.model.biases, audit.scores,
                          min(active, 9), active=active, winner=audit.predicted if active == 10 else None)
        self.status.set(f'Classificacao / somas calculadas: {active} de 10 / '
                        f'{"rede sem treino" if not self.records else "pesos atuais"}')
        if active == 10:
            self.classifying = False
            self.selected.set(str(audit.predicted))
            self.result.set(f'Digito: {audit.predicted}')
            self.refresh()
            return
        self.classification_active += 1
        self.job = self.root.after(max(60, int(self.delay.get())), self.classification_tick)

    def refresh(self):
        if self.classifying:
            return
        audit = self.model.inspect(self.features)
        active = 0 if self.pending and self.pending['phase'] == 'entrada' else 10
        target = self.pending['label'] if self.pending else None
        self.network.show(self.features, self.model.weights, self.model.biases, audit.scores,
                          int(self.selected.get()), active=active,
                          winner=audit.predicted if active == 10 else None, target=target)
        tab = self.tabs.select()
        if tab == str(self.audit_tab):
            self.refresh_audit()
        elif tab == str(self.metrics_tab):
            self.refresh_metrics()

    def refresh_audit(self):
        digit = int(self.selected.get())
        audit = self.model.inspect(self.features)
        current = self.model.weights[digit]
        baseline = self.snapshot_weights[digit]
        before = self.last_step.weights_before[digit] if self.last_step else current
        before_bias = self.last_step.biases_before[digit] if self.last_step else self.model.biases[digit]
        limit = max(float(np.max(np.abs([baseline, current]))), 0.01)
        delta = current - baseline
        delta_limit = max(float(np.max(np.abs(delta))), 0.01)
        for axis, values, title, bound in zip(
            self.audit_axes, (baseline, current, delta),
            ('Referencia', 'Pesos atuais', 'Delta da referencia'), (limit, limit, delta_limit)
        ):
            axis.clear()
            axis.imshow(values.reshape(8, 8), cmap='PiYG', vmin=-bound, vmax=bound)
            axis.set_title(f'{title}\n[{-bound:.3f}, {bound:.3f}]', fontsize=10)
            axis.set_xticks([])
            axis.set_yticks([])
        self.audit_canvas.draw_idle()
        contribution = audit.contributions[digit]
        self.score_label.set(f'Saida {digit}: soma(x * peso) {contribution.sum():+.5f} '
                             f'+ bias {audit.biases[digit]:+.5f} = {audit.scores[digit]:+.5f}')
        self.table.delete(*self.table.get_children())
        for index in np.argsort(-np.abs(contribution)):
            values = (self.features[index], baseline[index], current[index], delta[index],
                      before[index], current[index] - before[index], contribution[index])
            self.table.insert('', 'end', values=(f'{index // 8},{index % 8}',
                                                *(f'{value:+.5f}' for value in values)))
        bias = self.model.biases[digit]
        values = (1.0, self.snapshot_biases[digit], bias, bias - self.snapshot_biases[digit],
                  before_bias, bias - before_bias, bias)
        self.table.insert('', 'end', values=('bias', *(f'{value:+.5f}' for value in values)))

    def refresh_metrics(self):
        for axis in (self.accuracy_axis, self.error_axis, self.confusion_axis):
            axis.clear()
        epochs = [entry['epoch'] for entry in self.history]
        self.accuracy_axis.plot(epochs, [entry['train'] for entry in self.history], label='Treino', color='#087f70')
        self.accuracy_axis.plot(epochs, [entry['test'] for entry in self.history], label='Teste', color='#ca4260')
        self.accuracy_axis.set(title='Acuracia ao fim da epoca', xlabel='Epoca', ylim=(0, 1.05))
        self.accuracy_axis.legend(fontsize=8)
        self.error_axis.plot(epochs, [entry['mistakes'] for entry in self.history], color='#536ab8')
        self.error_axis.set(title='Erros antes da atualizacao', xlabel='Epoca', ylabel='Amostras')
        matrix = confusion_matrix(self.data.test_labels, self.model.predict(self.data.test_images), labels=range(10))
        self.confusion_axis.imshow(matrix, cmap='Greens')
        self.confusion_axis.set(title='Matriz de confusao / teste / pesos atuais',
                                xlabel='Previsto', ylabel='Real', xticks=range(10), yticks=range(10))
        for row in range(10):
            for column in range(10):
                self.confusion_axis.text(column, row, str(matrix[row, column]), ha='center', va='center',
                                         fontsize=7, color='white' if matrix[row, column] > matrix.max() / 2 else '#24332f')
        self.metrics_canvas.draw_idle()

    def capture(self):
        self.snapshot_weights = self.model.weights.copy()
        self.snapshot_biases = self.model.biases.copy()
        self.snapshot_updates = self.model.updates
        self.snapshot_label.set(f'Referencia: epoca {self.epoch}, {self.snapshot_updates} atualizacoes')
        self.refresh()

    def export_audit(self, filename):
        audit = self.model.inspect(self.features)
        path = Path(filename)
        if path.suffix.lower() == '.csv':
            with path.open('w', newline='', encoding='utf-8') as output:
                writer = csv.writer(output)
                writer.writerow(['class', 'pixel', 'input', 'reference_weight', 'current_weight',
                                 'delta_reference', 'before_last_step', 'delta_last_step',
                                 'contribution', 'score', 'predicted'])
                for digit in range(10):
                    for index in range(65):
                        is_bias = index == 64
                        value = 1.0 if is_bias else self.features[index]
                        weight = self.model.biases[digit] if is_bias else self.model.weights[digit, index]
                        baseline = self.snapshot_biases[digit] if is_bias else self.snapshot_weights[digit, index]
                        before = weight if self.last_step is None else (
                            self.last_step.biases_before[digit] if is_bias else self.last_step.weights_before[digit, index])
                        writer.writerow([digit, 'bias' if is_bias else index, value, baseline,
                                         weight, weight - baseline, before, weight - before,
                                         value * weight, audit.scores[digit], audit.predicted])
        else:
            payload = {
                'architecture': [64, 10], 'rule': 'multiclass perceptron, argmax, first index on ties',
                'seed': 42, 'normalization': 'pixel / 16', 'source': self.source,
                'learning_rate': self.model.learning_rate, 'epochs_completed': self.epoch,
                'updates': self.model.updates, 'features': self.features.tolist(),
                'weights': self.model.weights.tolist(), 'biases': self.model.biases.tolist(),
                'reference_weights': self.snapshot_weights.tolist(),
                'reference_biases': self.snapshot_biases.tolist(), 'reference_updates': self.snapshot_updates,
                'scores': audit.scores.tolist(), 'contributions': audit.contributions.tolist(),
                'predicted': audit.predicted, 'history': self.history, 'training_log': self.records,
                'train_indices': self.data.train_indices.tolist(), 'test_indices': self.data.test_indices.tolist(),
                'dataset_description': self.data.description,
            }
            path.write_text(json.dumps(payload, indent=2), encoding='utf-8')

    def export_dialog(self):
        self.pause()
        self.classifying = False
        filename = filedialog.asksaveasfilename(defaultextension='.json',
                    filetypes=[('Auditoria completa', '*.json'), ('Pesos e contribuicoes', '*.csv')])
        if filename:
            try:
                self.export_audit(filename)
                self.status.set(f'Auditoria exportada: {Path(filename).name}')
            except OSError as error:
                messagebox.showerror('Falha ao exportar', str(error))

    def reset(self):
        self.pause()
        self.classifying = False
        self.model = MultiClassPerceptron()
        self.random = np.random.default_rng(42)
        self.pending = self.last_step = self.order = None
        self.epoch = self.position = self.mistakes = 0
        self.history.clear()
        self.records.clear()
        self.features = self.drawing.copy()
        self.source = self.drawing_source
        self.rate_entry.configure(state='normal')
        self.epoch_entry.configure(state='normal')
        self.run_button.configure(text='Treinar')
        self.result.set('Resultado: --')
        self.progress.set('Epoca 0 | 0 atualizacoes')
        self.status.set('Rede reiniciada / pesos zerados')
        self.capture()

    def close(self):
        self.pause()
        self.root.destroy()


def main():
    root = tk.Tk()
    PerceptronApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()