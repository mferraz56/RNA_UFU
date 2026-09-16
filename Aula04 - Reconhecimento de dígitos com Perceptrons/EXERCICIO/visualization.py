import tkinter as tk

import numpy as np


class NetworkView(tk.Canvas):
    def __init__(self, parent):
        super().__init__(parent, background='#f7faf9', highlightthickness=0)
        self.state = None
        self.bind('<Configure>', lambda event: self.redraw())

    def show(self, features, weights, biases, scores, selected, active=10,
               winner=None, target=None, bipolar=False):
        self.state = (features.copy(), weights.copy(), biases.copy(), scores.copy(),
                      selected, active, winner, target, bipolar)
        self.redraw()

    def redraw(self):
        self.delete('all')
        if self.state is None:
            return
        features, weights, biases, scores, selected, active, winner, target, bipolar = self.state
        width, height = max(self.winfo_width(), 620), max(self.winfo_height(), 380)
        title = '64 entradas / bipolar (-1/+1)' if bipolar else '64 entradas / intensidade do pixel'
        self.create_text(25, 24, anchor='w', text=title,
                         font=('Bahnschrift', 12), fill='#24332f')
        self.create_text(width * 0.62, 24, anchor='w', text='10 saidas / soma linear',
                         font=('Bahnschrift', 12), fill='#24332f')
        spacing = min((width * 0.38 - 40) / 7, (height - 150) / 7)
        inputs = [(32 + (index % 8) * spacing, 90 + (index // 8) * spacing)
                  for index in range(64)]
        outputs = [(width * 0.66, 65 + digit * (height - 130) / 9)
                   for digit in range(10)]
        destination = outputs[selected]
        contributions = features * weights[selected]
        scale = max(float(np.max(np.abs(contributions))), 0.001)
        for index, origin in enumerate(inputs):
            value = contributions[index]
            color = '#087f70' if value > 0 else '#ca4260' if value < 0 else '#dce3e0'
            self.create_line(*origin, *destination, fill=color,
                             width=0.5 + 2.5 * abs(value) / scale, tags='connection')
        radius = max(5, min(13, spacing * 0.35))
        for index, (horizontal, vertical) in enumerate(inputs):
            intensity = (features[index] + 1) / 2 if bipolar else features[index]
            shade = int(245 - np.clip(intensity, 0, 1) * 210)
            self.create_oval(horizontal - radius, vertical - radius,
                             horizontal + radius, vertical + radius,
                             fill=f'#{shade:02x}{shade:02x}{shade:02x}',
                             outline='#b0beb8', tags='input')
            if bipolar:
                self.create_text(horizontal, vertical, text=f'{features[index]:+.0f}',
                                 fill='white' if intensity > 0.5 else '#24332f',
                                 font=('Consolas', 8), tags='input_value')
        for digit, (horizontal, vertical) in enumerate(outputs):
            color = '#087f70' if digit == winner else '#d8ece7' if digit < active else '#ffffff'
            outline = '#ca4260' if digit == target else '#087f70' if digit == selected else '#aab9b2'
            self.create_oval(horizontal - 14, vertical - 14, horizontal + 14, vertical + 14,
                             fill=color, outline=outline,
                             width=3 if digit in (selected, target) else 1, tags='output')
            self.create_text(horizontal, vertical, text=str(digit),
                             fill='white' if digit == winner else '#24332f',
                             font=('Consolas', 11, 'bold'))
            text = f'{scores[digit]:+.3f}  b={biases[digit]:+.2f}' if digit < active else '...'
            self.create_text(horizontal + 22, vertical, anchor='w', text=text,
                             font=('Consolas', 10), fill='#24332f')
        self.create_text(25, height - 37, anchor='w',
                         text=f'Saida {selected} | contribuicao positiva: verde | negativa: rosa',
                         fill='#53635c', font=('Bahnschrift', 10))
        self.create_text(25, height - 18, anchor='w',
                         text='Vencedor: preenchimento verde | rotulo esperado: contorno rosa',
                         fill='#53635c', font=('Bahnschrift', 10))