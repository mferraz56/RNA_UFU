import tkinter as tk
import numpy as np
from PIL import Image, ImageTk
import matplotlib.cm as cm

from Projeto_Solar_RNA.dataset_solar import CLASSES, CLASS_GROUPS, GROUP_COLORS

CLASSES_SHORT = [
    '0: No-Anomaly',
    '1: Cell',
    '2: Cell-Multi',
    '3: Cracking',
    '4: Hot-Spot',
    '5: Hot-Spot-M',
    '6: Shadowing',
    '7: Diode',
    '8: Diode-Multi',
    '9: Vegetation',
    '10: Soiling',
    '11: Offline'
]


class SolarNetworkView(tk.Canvas):
    """
    Tkinter Canvas component that visualizes:
    1. Spatial Input Neurons Matrix (24x40 pixels)
    2. Hidden Layer Feature Detectors Grid (4x6 = 24 neurons)
    3. Output Layer (12 Classes grouped into 4 Semantic Domains)
    4. Continuous Active Synaptic Edges (Input -> Hidden -> Output)
    """
    def __init__(self, parent, **kwargs):
        super().__init__(parent, background='#0f172a', highlightthickness=0, **kwargs)
        self.state = None
        self.hovered_pixel = None  # (row, col)
        self.selected_pixel = None  # (row, col)
        self.selected_hidden = None  # int index
        self.selected_output = 0     # int index
        
        # Geometry cache
        self.pixel_rects = {}   # (r, c) -> (x1, y1, x2, y2)
        self.hidden_nodes = {}  # h_idx -> (cx, cy)
        self.output_nodes = {}  # o_idx -> (cx, cy)
        
        self.bind('<Configure>', lambda event: self.redraw())
        self.bind('<Motion>', self.on_mouse_move)
        self.bind('<Button-1>', self.on_mouse_click)

    def update_view(self, features, weights=None, biases=None, scores=None, probs=None,
                    selected_output=0, selected_hidden=None, selected_pixel=None,
                    winner=None, target=None, is_mlp=False, hidden_activations=None,
                    hidden_grid_shape=(4, 6), show_edges=True):
        self.state = {
            'features': np.asarray(features, dtype=np.float32).reshape(40, 24) if len(features) == 960 else features,
            'weights': weights,
            'biases': biases,
            'scores': np.asarray(scores, dtype=np.float32) if scores is not None else None,
            'probs': np.asarray(probs, dtype=np.float32) if probs is not None else None,
            'winner': int(winner) if winner is not None else None,
            'target': int(target) if target is not None else None,
            'is_mlp': bool(is_mlp),
            'hidden_activations': np.asarray(hidden_activations, dtype=np.float32) if hidden_activations is not None else None,
            'hidden_grid_shape': hidden_grid_shape,
            'show_edges': bool(show_edges)
        }
        
        self.selected_output = int(selected_output)
        if selected_hidden is not None:
            self.selected_hidden = int(selected_hidden)
        if selected_pixel is not None:
            self.selected_pixel = selected_pixel

        self.redraw()

    def on_mouse_move(self, event):
        x, y = event.x, event.y
        found = None
        for (r, c), (x1, y1, x2, y2) in self.pixel_rects.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                found = (r, c)
                break
                
        if found != self.hovered_pixel:
            self.hovered_pixel = found
            self.redraw()

    def on_mouse_click(self, event):
        x, y = event.x, event.y
        for (r, c), (x1, y1, x2, y2) in self.pixel_rects.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.selected_pixel = (r, c)
                self.redraw()
                return

        for h_idx, (cx, cy) in self.hidden_nodes.items():
            if (x - cx)**2 + (y - cy)**2 <= 14**2:
                self.selected_hidden = h_idx
                self.redraw()
                return

        for o_idx, (cx, cy) in self.output_nodes.items():
            if (x - cx)**2 + (y - cy)**2 <= 16**2:
                self.selected_output = o_idx
                self.redraw()
                return

    def redraw(self):
        self.delete('all')
        if self.state is None:
            return

        st = self.state
        img_2d = st['features']  # (40, 24)
        scores = st['scores']
        probs = st['probs']
        winner = st['winner']
        target = st['target']
        is_mlp = st['is_mlp']
        hidden_acts = st['hidden_activations']
        h_rows, h_cols = st['hidden_grid_shape']

        width = max(self.winfo_width(), 750)
        height = max(self.winfo_height(), 460)

        # Title / Mode banner
        title_text = "Painel Espacial: Matriz 24x40 (960 Neurônios Entrada)"
        if is_mlp:
            title_text += f" ➔ Grade Oculta {h_rows}x{h_cols} ({h_rows*h_cols} Detectores) ➔ 12 Classes"
        else:
            title_text += " ➔ 12 Neurônios de Saída (Perceptron)"

        self.create_text(20, 18, anchor='w', text=title_text,
                         font=('Bahnschrift', 11, 'bold'), fill='#38bdf8')

        margin_top = 45
        margin_bottom = 35
        avail_h = height - margin_top - margin_bottom

        # 1. DRAW INPUT SPATIAL MATRIX 24x40 (Left side)
        mat_x = 25
        mat_y = margin_top
        cell_w = min(8.0, (width * 0.28) / 24)
        cell_h = min(9.5, avail_h / 40)

        self.pixel_rects.clear()
        
        for r in range(40):
            for c in range(24):
                val = img_2d[r, c] if (r < img_2d.shape[0] and c < img_2d.shape[1]) else 0.0
                val_norm = np.clip(val, 0.0, 1.0)
                
                rgb = cm.inferno(val_norm, bytes=True)
                fill_hex = f'#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}'
                
                x1 = mat_x + c * cell_w
                y1 = mat_y + r * cell_h
                x2 = x1 + cell_w
                y2 = y1 + cell_h
                
                self.pixel_rects[(r, c)] = (x1, y1, x2, y2)
                
                is_active_pix = (self.hovered_pixel == (r, c)) or (self.selected_pixel == (r, c))
                outline_color = '#00f0ff' if is_active_pix else '#1e293b'
                lw = 2 if is_active_pix else 1
                
                self.create_rectangle(x1, y1, x2, y2, fill=fill_hex, outline=outline_color, width=lw)

        self.create_text(mat_x, mat_y + 40 * cell_h + 12, anchor='w',
                         text=f"Matriz IR 24x40 ({24*40} Pixels)",
                         font=('Bahnschrift', 9), fill='#94a3b8')

        # 2. DRAW HIDDEN FEATURE DETECTORS GRID (Middle)
        self.hidden_nodes.clear()
        if is_mlp and hidden_acts is not None:
            hid_x_start = width * 0.38
            hid_y_start = margin_top + 40
            
            num_h = len(hidden_acts)
            spacing_x = min(38.0, (width * 0.22) / max(h_cols, 1))
            spacing_y = min(42.0, (avail_h - 60) / max(h_rows, 1))
            
            for hr in range(h_rows):
                for hc in range(h_cols):
                    h_idx = hr * h_cols + hc
                    if h_idx < num_h:
                        cx = hid_x_start + hc * spacing_x
                        cy = hid_y_start + hr * spacing_y
                        self.hidden_nodes[h_idx] = (cx, cy)

        # 3. DRAW OUTPUT CLASS NODES (Right side)
        self.output_nodes.clear()
        out_x = width - 190
        out_spacing = avail_h / 12.0
        
        for i in range(12):
            cy = margin_top + i * out_spacing + 10
            self.output_nodes[i] = (out_x, cy)

        # 4. DRAW CONTINUOUS SYNAPTIC EDGES (Input -> Hidden -> Output)
        if st.get('show_edges', True):
            # A) Input -> Hidden Edges (Always visible & active)
        if is_mlp and len(self.hidden_nodes) > 0:
            # Connect spatial sub-regions of 24x40 matrix to the 4x6 hidden detectors
            for hr in range(h_rows):
                for hc in range(h_cols):
                    h_idx = hr * h_cols + hc
                    if h_idx in self.hidden_nodes:
                        hcx, hcy = self.hidden_nodes[h_idx]
                        act = hidden_acts[h_idx] if h_idx < len(hidden_acts) else 0.0
                        
                        # Find corresponding spatial sub-region in 24x40 matrix
                        r_start = int((hr / h_rows) * 40)
                        r_end = int(((hr + 1) / h_rows) * 40)
                        c_start = int((hc / h_cols) * 24)
                        c_end = int(((hc + 1) / h_cols) * 24)
                        
                        # Center of region in canvas
                        region_cx = mat_x + ((c_start + c_end) / 2.0) * cell_w
                        region_cy = mat_y + ((r_start + r_end) / 2.0) * cell_h
                        
                        # Line color & width based on activation level
                        if act > 0.1:
                            col = '#0284c7' if act < 0.5 else '#38bdf8'
                            lw = 0.8 + 2.0 * np.clip(act, 0, 1)
                        else:
                            col = '#1e293b'
                            lw = 0.5

                        self.create_line(region_cx, region_cy, hcx, hcy,
                                         fill=col, width=lw, tags='edge_layer1')

        # B) Hidden -> Output Edges (Always visible for Active/Selected/Winner Outputs)
        if is_mlp and len(self.hidden_nodes) > 0:
            target_outputs = set()
            if winner is not None:
                target_outputs.add(winner)
            target_outputs.add(self.selected_output)
            
            for o_idx in target_outputs:
                ocx, ocy = self.output_nodes[o_idx]
                is_win_edge = (o_idx == winner)
                
                for h_idx, (hcx, hcy) in self.hidden_nodes.items():
                    act = hidden_acts[h_idx] if h_idx < len(hidden_acts) else 0.0
                    if act > 0.05:
                        edge_col = '#2ecc71' if is_win_edge else '#0284c7'
                        lw = 0.8 + 2.5 * np.clip(act, 0, 1)
                        self.create_line(hcx, hcy, ocx, ocy, fill=edge_col, width=lw, tags='edge_layer2')

        # C) Specific Hovered / Selected Pixel Edge Streamer (High-Contrast Highlight)
        active_pix = self.hovered_pixel or self.selected_pixel
        if active_pix and (active_pix in self.pixel_rects):
            r, c = active_pix
            px1, py1, px2, py2 = self.pixel_rects[(r, c)]
            pix_center = ((px1 + px2) / 2, (py1 + py2) / 2)
            
            if is_mlp and len(self.hidden_nodes) > 0:
                for h_idx, h_pos in self.hidden_nodes.items():
                    self.create_line(pix_center[0], pix_center[1], h_pos[0], h_pos[1],
                                     fill='#00f0ff', width=1.8, tags='highlight_edge')
            else:
                dest = self.output_nodes[self.selected_output]
                self.create_line(pix_center[0], pix_center[1], dest[0], dest[1],
                                 fill='#00f0ff', width=2.5, tags='highlight_edge')

        # D) Specific Hovered / Selected Hidden Node Edge Streamer
        if is_mlp and self.selected_hidden is not None and (self.selected_hidden in self.hidden_nodes):
            hcx, hcy = self.hidden_nodes[self.selected_hidden]
            dest = self.output_nodes[self.selected_output]
            self.create_line(hcx, hcy, dest[0], dest[1],
                             fill='#f59e0b', width=3.0, tags='highlight_edge')

        # 5. RENDER HIDDEN NODES (Ovals)
        if is_mlp:
            for h_idx, (cx, cy) in self.hidden_nodes.items():
                act = hidden_acts[h_idx] if (hidden_acts is not None and h_idx < len(hidden_acts)) else 0.0
                is_sel = (h_idx == self.selected_hidden)
                
                intensity = np.clip(act, 0, 1)
                g_val = int(120 + intensity * 135)
                fill_hex = f'#00{g_val:02x}80' if intensity > 0.05 else '#1e293b'
                outline_color = '#f59e0b' if is_sel else '#38bdf8' if intensity > 0.2 else '#475569'
                
                self.create_oval(cx - 11, cy - 11, cx + 11, cy + 11,
                                 fill=fill_hex, outline=outline_color, width=2 if is_sel else 1)
                
                self.create_text(cx, cy, text=f"H{h_idx}", fill='white' if intensity > 0.2 else '#94a3b8',
                                 font=('Consolas', 7, 'bold'))

            if len(self.hidden_nodes) > 0:
                first_h = self.hidden_nodes[0]
                self.create_text(first_h[0], margin_top - 12, anchor='center',
                                 text=f"Detectores Ocultos ({h_rows}x{h_cols})",
                                 font=('Bahnschrift', 9, 'bold'), fill='#38bdf8')

        # 6. RENDER OUTPUT NODES (12 Classes)
        for digit, (cx, cy) in self.output_nodes.items():
            is_winner = (digit == winner)
            is_target = (digit == target)
            is_selected = (digit == self.selected_output)
            
            fill_hex = '#2ecc71' if is_winner else '#1e293b'
            outline_hex = '#e74c3c' if is_target else '#38bdf8' if is_selected else '#475569'
            
            self.create_oval(cx - 13, cy - 13, cx + 13, cy + 13,
                             fill=fill_hex, outline=outline_hex, width=3 if (is_selected or is_target) else 1)
            
            self.create_text(cx, cy, text=str(digit), fill='white' if is_winner else '#f8fafc',
                             font=('Consolas', 9, 'bold'))
            
            pr = probs[digit] * 100.0 if (probs is not None and digit < len(probs)) else 0.0
            lbl_txt = f"{CLASSES_SHORT[digit]} ({pr:.1f}%)"
            
            self.create_text(cx + 18, cy, anchor='w', text=lbl_txt,
                             font=('Consolas', 9), fill='#f8fafc' if is_winner else '#cbd5e1')

        # Footer Legend
        pix_info = f"Pixel Selecionado: {self.hovered_pixel or self.selected_pixel or 'Passe o mouse'}"
        self.create_text(20, height - 16, anchor='w',
                         text=f"{pix_info} | Conexão Ativa: Azul/Verde | Vencedor: Verde Cheio | Alvo Real: Contorno Vermelho",
                         fill='#94a3b8', font=('Bahnschrift', 9))
