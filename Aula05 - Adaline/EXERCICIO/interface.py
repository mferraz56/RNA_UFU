import tkinter as tk
from tkinter import ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class TrainingWindow:
    def __init__(self, root, model, features, targets, epochs):
        self.root, self.model = root, model
        self.features, self.targets, self.epochs = features, targets, epochs
        self.initial_weights, self.initial_bias = model.weights.copy(), model.bias
        self.job = None
        self.running = self.completed = self.accepted = self.failed = False
        self.history = []
        self.last_step = None
        self.iterator = model.iter_fit(features, targets, epochs)
        self.mode = tk.StringVar(value='Amostra')
        self.delay = tk.DoubleVar(value=150)
        self.status = tk.StringVar(value='Pronto / pesos iniciais')
        self.details = tk.StringVar(value='')
        root.title('Trabalho 5 / Adaline / Treinamento ao vivo')
        root.geometry('1180x880')
        root.minsize(1100, 820)
        style = ttk.Style(root)
        style.theme_use('clam')
        style.configure('.', font=('Bahnschrift', 10))
        frame = ttk.Frame(root, padding=14)
        frame.pack(fill='both', expand=True)
        ttk.Label(frame, text='ADALINE / treinamento ao vivo', font=('Bahnschrift', 19, 'bold')).pack(anchor='w')
        ttk.Label(frame, text=f'{len(features)} amostras de treino | {epochs} epocas | '
                  f'taxa {model.learning_rate} | 2 entradas + bias').pack(anchor='w', pady=(4, 10))
        controls = ttk.Frame(frame)
        controls.pack(fill='x')
        self.play_button = ttk.Button(controls, text='Treinar', command=self.toggle)
        self.play_button.pack(side='left')
        self.step_button = ttk.Button(controls, text='Passo', command=self.single_step)
        self.step_button.pack(side='left', padx=6)
        ttk.Button(controls, text='Reiniciar', command=self.reset).pack(side='left')
        ttk.Label(controls, text='Avanco').pack(side='left', padx=(16, 4))
        ttk.Combobox(controls, textvariable=self.mode, values=['Amostra', 'Epoca'],
                     state='readonly', width=9).pack(side='left')
        ttk.Label(controls, text='Intervalo (ms)').pack(side='left', padx=(16, 4))
        ttk.Scale(controls, from_=20, to=1000, variable=self.delay, length=160).pack(side='left')
        self.finish_button = ttk.Button(controls, text='Testar e salvar', command=self.accept, state='disabled')
        self.finish_button.pack(side='right')
        ttk.Label(frame, textvariable=self.status).pack(anchor='w', pady=(10, 4))
        self.network = tk.Canvas(frame, height=210, background='#f7faf9', highlightthickness=0)
        self.network.pack(fill='x')
        self.network.bind('<Configure>', lambda event: self.draw_network())
        ttk.Label(frame, textvariable=self.details, font=('Consolas', 10),
                  wraplength=1050).pack(anchor='w', pady=8)
        self.figure = Figure(figsize=(10, 3.6), dpi=100, layout='constrained')
        self.error_axis, self.boundary_axis = self.figure.subplots(1, 2)
        self.error_axis.set(title='Erro quadratico total / treino', xlabel='Epoca', ylabel='Soma dos erros ao quadrado')
        self.online_line, = self.error_axis.plot([], [], label='Acumulado nas atualizacoes', color='#c64b64')
        initial_sse = np.sum((targets - model.decision_function(features)) ** 2)
        self.initial_sse = float(initial_sse)
        self.final_line, = self.error_axis.plot([0], [initial_sse], label='Pesos fixos ao fim da epoca', color='#087f70')
        self.error_axis.legend(fontsize=8)
        self.error_axis.grid(alpha=0.25)
        self.lower, self.upper = features.min(axis=0) - 0.3, features.max(axis=0) + 0.3
        self.boundary_axis.set(title='Fronteira / pesos atuais', xlabel='s1', ylabel='s2',
                               xlim=(self.lower[0], self.upper[0]), ylim=(self.lower[1], self.upper[1]))
        for target, color in ((-1, '#c64b64'), (1, '#087f70')):
            points = features[targets == target]
            self.boundary_axis.scatter(points[:, 0], points[:, 1], color=color, label=f'Classe {target:+d}')
        self.boundary_line, = self.boundary_axis.plot([], [], color='#394b9a', label='Saida linear = 0')
        self.current_point = self.boundary_axis.scatter([], [], s=180, facecolors='none',
                                                       edgecolors='#202020', linewidths=2, label='Amostra atual')
        self.boundary_axis.legend(fontsize=8)
        self.boundary_axis.grid(alpha=0.25)
        self.canvas = FigureCanvasTkAgg(self.figure, master=frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        root.protocol('WM_DELETE_WINDOW', self.close)
        self.refresh()

    def toggle(self):
        if self.running:
            self.pause()
        elif not self.completed and not self.failed:
            self.running = True
            self.play_button.configure(text='Pausar')
            self.tick()

    def pause(self):
        self.running = False
        if self.job is not None:
            self.root.after_cancel(self.job)
            self.job = None
        self.play_button.configure(text='Continuar')

    def tick(self):
        self.job = None
        if not self.running:
            return
        self.advance()
        if self.running:
            self.job = self.root.after(int(self.delay.get()), self.tick)

    def single_step(self):
        self.pause()
        self.advance()

    def advance(self):
        if self.completed or self.failed:
            return
        try:
            while True:
                step = next(self.iterator)
                self.last_step = step
                if step['epoch_result'] is not None:
                    self.history.append(step['epoch_result'])
                if len(self.history) == self.epochs:
                    self.completed = True
                    self.pause()
                    self.play_button.configure(state='disabled')
                    self.step_button.configure(state='disabled')
                    self.finish_button.configure(state='normal')
                    break
                if self.mode.get() == 'Amostra' or step['epoch_result'] is not None:
                    break
            self.refresh()
        except (ValueError, StopIteration) as error:
            self.failed = True
            self.pause()
            self.play_button.configure(state='disabled')
            self.step_button.configure(state='disabled')
            self.status.set(f'Falha no treino: {error}. Reinicie para tentar novamente.')

    def refresh(self):
        step = self.last_step
        if step is None:
            self.details.set(f'Pesos iniciais: {self.model.weights} | bias: {self.model.bias:+.6f}\n'
                             f'EQT inicial: {self.initial_sse:.6f}')
        else:
            self.status.set(f'{"Concluido" if self.completed else "Treino"} / epoca {step["epoch"]}/{self.epochs} '
                            f'/ amostra {step["sample_index"] + 1}/{len(self.features)}')
            delta = step['weights_after'] - step['weights_before']
            self.details.set(f'Antes: w1={step["weights_before"][0]:+.6f}, w2={step["weights_before"][1]:+.6f}, '
                             f'b={step["bias_before"]:+.6f} | u={step["linear_output"]:+.6f}, '
                             f't={step["target"]:+.0f}, erro=t-u={step["error"]:+.6f}\n'
                             f'Depois: w1={self.model.weights[0]:+.6f}, w2={self.model.weights[1]:+.6f}, '
                             f'b={self.model.bias:+.6f} | delta w=({delta[0]:+.6f}, {delta[1]:+.6f}), '
                             f'delta b={self.model.bias - step["bias_before"]:+.6f}')
            self.current_point.set_offsets([step['features']])
        epochs = [entry['epoch'] for entry in self.history]
        self.online_line.set_data(epochs, [entry['online_sse'] for entry in self.history])
        self.final_line.set_data([0] + epochs, [self.initial_sse] + [entry['sse'] for entry in self.history])
        self.error_axis.relim()
        self.error_axis.autoscale_view()
        first_weight, second_weight = self.model.weights
        if abs(second_weight) >= abs(first_weight) and abs(second_weight) > 1e-12:
            horizontal = np.array([self.lower[0], self.upper[0]])
            self.boundary_line.set_data(horizontal, -(horizontal * first_weight + self.model.bias) / second_weight)
        elif abs(first_weight) > 1e-12:
            vertical = np.array([self.lower[1], self.upper[1]])
            self.boundary_line.set_data(-(vertical * second_weight + self.model.bias) / first_weight, vertical)
        else:
            self.boundary_line.set_data([], [])
        self.draw_network()
        self.canvas.draw_idle()

    def draw_network(self):
        self.network.delete('all')
        width = max(self.network.winfo_width(), 900)
        step = self.last_step
        features = step['features'] if step else np.zeros(2)
        weights = step['weights_before'] if step else self.model.weights
        bias = step['bias_before'] if step else self.model.bias
        center = (width * 0.52, 105)
        self.network.create_text(width / 2, 15, text='Propagacao da amostra / valores antes da atualizacao',
                                 font=('Bahnschrift', 11), fill='#24332f')
        for index, (name, value, weight) in enumerate(zip(('s1', 's2', 'bias'),
                                                        (*features, 1.0), (*weights, bias))):
            vertical = 55 + index * 60
            contribution = value * weight
            color = '#087f70' if contribution >= 0 else '#c64b64'
            self.network.create_line(110, vertical, *center, fill=color, width=2, arrow=tk.LAST)
            self.network.create_oval(60, vertical - 22, 110, vertical + 22,
                                     fill='#d8ece7' if step else '#ffffff', outline=color, tags='input')
            self.network.create_text(85, vertical, text=name, font=('Consolas', 11))
            label_y = vertical + 20 if index == 2 else vertical - 20
            self.network.create_text(140, label_y, anchor='w', text=f'{value:+.3f} * {weight:+.3f} = {contribution:+.3f}',
                                     font=('Consolas', 10), fill=color)
        self.network.create_oval(center[0] - 37, center[1] - 30, center[0] + 37, center[1] + 30,
                                 fill='#087f70' if step else '#ffffff', outline='#087f70', tags='neuron')
        self.network.create_text(*center, text='LINEAR', fill='white' if step else '#24332f', font=('Consolas', 11))
        self.network.create_line(center[0] + 37, center[1], width * 0.76, center[1], arrow=tk.LAST, fill='#394b9a', width=2)
        output = f'u = {step["linear_output"]:+.5f}\nt = {step["target"]:+.0f}\nerro = {step["error"]:+.5f}' if step else 'u = --\nt = --\nerro = --'
        self.network.create_text(width * 0.86, center[1], text=output, font=('Consolas', 12), fill='#24332f')

    def reset(self):
        self.pause()
        self.iterator.close()
        self.model.weights[:] = self.initial_weights
        self.model.bias = self.initial_bias
        self.iterator = self.model.iter_fit(self.features, self.targets, self.epochs)
        self.history.clear()
        self.last_step = None
        self.completed = self.accepted = self.failed = False
        self.play_button.configure(text='Treinar', state='normal')
        self.step_button.configure(state='normal')
        self.finish_button.configure(state='disabled')
        self.current_point.set_offsets(np.empty((0, 2)))
        self.status.set('Reiniciado / pesos iniciais')
        self.refresh()

    def accept(self):
        if self.completed:
            self.accepted = True
            self.close()

    def close(self):
        self.pause()
        self.iterator.close()
        self.root.quit()


def run_training(model, features, targets, epochs):
    root = tk.Tk()
    try:
        window = TrainingWindow(root, model, features, targets, epochs)
        root.mainloop()
        return window.history if window.accepted else None
    finally:
        root.destroy()