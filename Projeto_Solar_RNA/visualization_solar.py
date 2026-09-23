import tkinter as tk
import numpy as np
from PIL import Image, ImageTk
import matplotlib.cm as cm

from Projeto_Solar_RNA.dataset_solar import CLASSES, CLASS_GROUPS, GROUP_COLORS
from Projeto_Solar_RNA.network_config import load_network_config

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
    2. Configured hidden layer grid and input block means
    3. Output Layer (12 Classes grouped into 4 Semantic Domains)
    4. Continuous Active Synaptic Edges (Input -> Hidden -> Output)
    """
    def __init__(self, parent, config=None, **kwargs):
        super().__init__(parent, background='#0f172a', highlightthickness=0, **kwargs)
        self.config = config if config is not None else load_network_config()
        self.state = None
        self.hovered_pixel = None  # (row, col)
        self.selected_pixel = None  # (row, col)
        self.selected_hidden = None  # int index
        self.selected_output = 0     # int index
        
        # Geometry cache
        self.pixel_rects = {}   # (r, c) -> (x1, y1, x2, y2)
        self.hidden_nodes = {}  # h_idx -> (cx, cy)
        self.hidden_radius = 11
        self.hidden_bounds = None
        self.hovered_hidden = None
        self.hidden_image = None
        self.output_nodes = {}  # o_idx -> (cx, cy)
        
        self.bind('<Configure>', lambda event: self.redraw())
        self.bind('<Motion>', self.on_mouse_move)
        self.bind('<Button-1>', self.on_mouse_click)
        self.bind('<Leave>', self.on_mouse_leave)

    def update_view(self, features, weights=None, biases=None, scores=None, probs=None,
                    selected_output=0, selected_hidden=None, selected_pixel=None,
                    winner=None, target=None, is_mlp=False, hidden_activations=None,
                    hidden_grid_shape=None, show_edges=True):
        feature_vector = np.asarray(features, dtype=np.float32).ravel()
        if feature_vector.size != self.config.num_inputs:
            raise ValueError(f'Esperado vetor de {self.config.num_inputs} entradas')
        hidden_grid_shape = self.config.hidden_grid if hidden_grid_shape is None else hidden_grid_shape
        self.state = {
            'features': feature_vector[:960].reshape(40, 24),
            'row_means': feature_vector[960:1000],
            'column_means': feature_vector[1000:1024],
            'mask_means': feature_vector[1024:].reshape(self.config.mask_grid),
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

    def hidden_at(self, x, y):
        if self.hidden_bounds is not None:
            left, top, cell_size, rows, columns = self.hidden_bounds
            if left <= x < left + columns * cell_size and top <= y < top + rows * cell_size:
                return int((y - top) // cell_size) * columns + int((x - left) // cell_size)
        for index, (center_x, center_y) in self.hidden_nodes.items():
            if (x - center_x)**2 + (y - center_y)**2 <= self.hidden_radius**2:
                return index
        return None

    def on_mouse_leave(self, event):
        self.hovered_hidden = None
        self.hovered_pixel = None
        self.redraw()

    def on_mouse_move(self, event):
        x, y = self.canvasx(event.x), self.canvasy(event.y)
        found = None
        for (r, c), (x1, y1, x2, y2) in self.pixel_rects.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                found = (r, c)
                break
                
        hidden = self.hidden_at(x, y)
        if found != self.hovered_pixel or hidden != self.hovered_hidden:
            self.hovered_pixel = found
            self.hovered_hidden = hidden
            self.redraw()

    def on_mouse_click(self, event):
        x, y = self.canvasx(event.x), self.canvasy(event.y)
        hidden = self.hidden_at(x, y)
        if hidden is not None:
            self.selected_hidden = hidden
            self.redraw()
            return
        for (r, c), (x1, y1, x2, y2) in self.pixel_rects.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.selected_pixel = (r, c)
                self.redraw()
                return

        for h_idx, (cx, cy) in self.hidden_nodes.items():
            if (x - cx)**2 + (y - cy)**2 <= self.hidden_radius**2:
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

        dense_hidden = h_rows * h_cols > 256
        width = max(self.winfo_width(), 1000, 530 + h_cols * 32 if not dense_hidden else 1000)
        mask_grid_rows, mask_grid_columns = self.config.mask_grid
        height = max(self.winfo_height(), 600 + mask_grid_rows * 28)
        if is_mlp and not dense_hidden:
            height = max(height, 180 + h_rows * 30)
        self.configure(scrollregion=(0, 0, width, height))

        # Title / Mode banner
        title_text = f"Entrada: {self.config.num_inputs}"
        if is_mlp:
            title_text += f" ➔ Oculta: {h_rows*h_cols} ({h_rows}x{h_cols})"
        else:
            title_text += " ➔ Perceptron"
        title_text += f" ➔ Saída: {self.config.output_grid[0]}x{self.config.output_grid[1]}"

        self.create_text(20, 18, anchor='w', text=title_text,
                         font=('Bahnschrift', 11, 'bold'), fill='#38bdf8')

        margin_top = 65
        margin_bottom = 70
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

        for row_index, value in enumerate(st['row_means']):
            rgb = cm.inferno(float(np.clip(value, 0, 1)), bytes=True)
            mean_x = mat_x + 24 * cell_w + 4
            mean_y = mat_y + row_index * cell_h
            self.create_rectangle(mean_x, mean_y, mean_x + cell_w, mean_y + cell_h,
                                  fill=f'#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}', outline='#1e293b')
        for column_index, value in enumerate(st['column_means']):
            rgb = cm.inferno(float(np.clip(value, 0, 1)), bytes=True)
            mean_x = mat_x + column_index * cell_w
            mean_y = mat_y + 40 * cell_h + 4
            self.create_rectangle(mean_x, mean_y, mean_x + cell_w, mean_y + 6,
                                  fill=f'#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}', outline='#1e293b')

        self.create_text(mat_x, mat_y + 40 * cell_h + 20, anchor='w',
                         text="960 pixels + 40 médias L + 24 médias C",
                         font=('Bahnschrift', 9), fill='#94a3b8')

        mask_rows, mask_columns = self.config.mask_shape
        for grid_row in range(mask_grid_rows):
            for grid_column in range(mask_grid_columns):
                left = mat_x + grid_column * mask_columns * cell_w
                top = mat_y + grid_row * mask_rows * cell_h
                self.create_rectangle(left, top, left + mask_columns * cell_w, top + mask_rows * cell_h,
                                      outline='#38bdf8', tags='mask_boundary')
        masks_top = mat_y + 40 * cell_h + 70
        self.create_text(mat_x, masks_top - 20, anchor='w',
                         text=f'{mask_grid_rows * mask_grid_columns} médias | máscaras {mask_rows}x{mask_columns}',
                         font=('Bahnschrift', 10), fill='#38bdf8')
        mask_width = min(60, 240 / mask_grid_columns)
        for (grid_row, grid_column), value in np.ndenumerate(st['mask_means']):
            left = mat_x + grid_column * mask_width
            top = masks_top + grid_row * 28
            rgb = cm.inferno(float(np.clip(value, 0, 1)), bytes=True)
            self.create_rectangle(left, top, left + mask_width, top + 28,
                                  fill=f'#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}',
                                  outline='#334155', tags='mask_cell')
            self.create_text(left + mask_width / 2, top + 14, text=f'{value:.2f}',
                             font=('Consolas', 8), fill='#000000' if value > 0.65 else '#ffffff',
                             tags='mask_value')

        # 2. DRAW HIDDEN FEATURE DETECTORS GRID (Middle)
        self.hidden_nodes.clear()
        self.hidden_bounds = None
        if is_mlp and hidden_acts is not None:
            hid_x_start = width * 0.38
            hid_y_start = margin_top + 40
            
            num_h = len(hidden_acts)
            spacing_x = min(38.0, (width * 0.28) / max(h_cols, 1))
            spacing_y = min(42.0, (avail_h - 60) / max(h_rows, 1))
            self.hidden_radius = min(11, spacing_x * 0.42, spacing_y * 0.42)
            if dense_hidden:
                grid_left = 290
                grid_top = margin_top
                cell_size = max(1, int(min((width - 530) / h_cols, (avail_h - 35) / h_rows)))
                spacing_x = spacing_y = cell_size
                hid_x_start = grid_left + cell_size / 2
                hid_y_start = grid_top + cell_size / 2
                self.hidden_bounds = (grid_left, grid_top, cell_size, h_rows, h_cols)
            
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
        out_spacing = min(40, avail_h / self.config.num_classes)
        self.create_text(out_x, margin_top - 16, anchor='w',
                 text=f'Saída {self.config.output_grid[0]}x{self.config.output_grid[1]}',
                 font=('Bahnschrift', 10, 'bold'), fill='#38bdf8')
        
        for i in range(self.config.num_classes):
            cy = margin_top + i * out_spacing + 10
            self.output_nodes[i] = (out_x, cy)

        # Compute max hidden activation for relative scaling
        max_h_act = max(float(np.max(hidden_acts)), 1e-4) if (hidden_acts is not None and len(hidden_acts) > 0) else 1.0

        # 4. DRAW CONTINUOUS SYNAPTIC EDGES (Input -> Hidden -> Output)
        if st.get('show_edges', True) and not dense_hidden:
            # A) Input -> Hidden Edges (Always visible & active)
            if is_mlp and len(self.hidden_nodes) > 0:
                for hr in range(h_rows):
                    for hc in range(h_cols):
                        h_idx = hr * h_cols + hc
                        if h_idx in self.hidden_nodes:
                            hcx, hcy = self.hidden_nodes[h_idx]
                            act = hidden_acts[h_idx] if h_idx < len(hidden_acts) else 0.0
                            rel_act = act / max_h_act
                            
                            r_start = int((hr / h_rows) * 40)
                            r_end = int(((hr + 1) / h_rows) * 40)
                            c_start = int((hc / h_cols) * 24)
                            c_end = int(((hc + 1) / h_cols) * 24)
                            
                            region_cx = mat_x + ((c_start + c_end) / 2.0) * cell_w
                            region_cy = mat_y + ((r_start + r_end) / 2.0) * cell_h
                            
                            if rel_act > 0.05:
                                col = '#0284c7' if rel_act < 0.5 else '#38bdf8'
                                lw = 0.8 + 2.2 * np.clip(rel_act, 0, 1)
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
                        rel_act = act / max_h_act
                        if rel_act > 0.05:
                            edge_col = '#2ecc71' if is_win_edge else '#0284c7'
                            lw = 0.8 + 2.5 * np.clip(rel_act, 0, 1)
                            self.create_line(hcx, hcy, ocx, ocy, fill=edge_col, width=lw, tags='edge_layer2')

        # C) Specific Hovered / Selected Pixel Edge Streamer (High-Contrast Highlight)
        active_pix = self.hovered_pixel or self.selected_pixel
        if st.get('show_edges', True) and active_pix and (active_pix in self.pixel_rects):
            r, c = active_pix
            px1, py1, px2, py2 = self.pixel_rects[(r, c)]
            pix_center = ((px1 + px2) / 2, (py1 + py2) / 2)
            
            if is_mlp and len(self.hidden_nodes) > 0:
                for h_idx, h_pos in self.hidden_nodes.items():
                    if dense_hidden and h_idx != self.selected_hidden:
                        continue
                    self.create_line(pix_center[0], pix_center[1], h_pos[0], h_pos[1],
                                     fill='#00f0ff', width=1.8, tags='highlight_edge')
            else:
                dest = self.output_nodes[self.selected_output]
                self.create_line(pix_center[0], pix_center[1], dest[0], dest[1],
                                 fill='#00f0ff', width=2.5, tags='highlight_edge')

        # D) Specific Hovered / Selected Hidden Node Edge Streamer
        if st.get('show_edges', True) and is_mlp and self.selected_hidden in self.hidden_nodes:
            hcx, hcy = self.hidden_nodes[self.selected_hidden]
            dest = self.output_nodes[self.selected_output]
            self.create_line(hcx, hcy, dest[0], dest[1],
                             fill='#f59e0b', width=3.0, tags='highlight_edge')

        # 5. RENDER HIDDEN NODES (Ovals)
        if is_mlp and not dense_hidden:
            for h_idx, (cx, cy) in self.hidden_nodes.items():
                act = hidden_acts[h_idx] if (hidden_acts is not None and h_idx < len(hidden_acts)) else 0.0
                is_sel = (h_idx == self.selected_hidden)
                rel_act = act / max_h_act
                
                g_val = int(120 + np.clip(rel_act, 0, 1) * 135)
                fill_hex = f'#00{g_val:02x}80' if act > 0.001 else '#1e293b'
                outline_color = '#f59e0b' if is_sel else '#00f0ff' if rel_act > 0.3 else '#475569'
                
                self.create_oval(cx - self.hidden_radius, cy - self.hidden_radius,
                                 cx + self.hidden_radius, cy + self.hidden_radius,
                                 fill=fill_hex, outline=outline_color, width=2 if is_sel else 1)
                
                self.create_text(cx, cy, text=f"H{h_idx}", fill='white' if act > 0.001 else '#94a3b8',
                                 font=('Consolas', 7, 'bold'))

            inspected = self.hovered_hidden if self.hovered_hidden in self.hidden_nodes else self.selected_hidden
            if inspected in self.hidden_nodes:
                center_x, center_y = self.hidden_nodes[inspected]
                self.create_text(center_x + 22, center_y, anchor='w',
                                 text=f'H{inspected}: {hidden_acts[inspected]:.5f}',
                                 font=('Consolas', 9), fill='#f59e0b', tags='hidden_detail')

            if len(self.hidden_nodes) > 0:
                first_h = self.hidden_nodes[0]
                self.create_text(first_h[0], margin_top - 12, anchor='center',
                                 text=f"Detectores Ocultos ({h_rows}x{h_cols})",
                                 font=('Bahnschrift', 9, 'bold'), fill='#38bdf8')

        if is_mlp and dense_hidden and self.hidden_bounds is not None:
            grid_left, grid_top, cell_size, rows, columns = self.hidden_bounds
            activation_grid = hidden_acts.reshape(rows, columns)
            colors = cm.viridis(np.clip(activation_grid / max_h_act, 0, 1), bytes=True)
            bitmap = Image.fromarray(colors).resize(
                (columns * cell_size, rows * cell_size), Image.Resampling.NEAREST)
            self.hidden_image = ImageTk.PhotoImage(bitmap, master=self)
            self.create_image(grid_left, grid_top, anchor='nw', image=self.hidden_image, tags='hidden_heatmap')
            self.create_text(grid_left, grid_top - 16, anchor='w',
                             text=f"Oculta {rows}x{columns} | {rows * columns} neurônios",
                             font=('Bahnschrift', 10, 'bold'), fill='#38bdf8')
            for hidden_index, color in ((self.selected_hidden, '#f59e0b'), (self.hovered_hidden, '#ffffff')):
                if hidden_index in self.hidden_nodes:
                    center_x, center_y = self.hidden_nodes[hidden_index]
                    self.create_rectangle(center_x - cell_size / 2, center_y - cell_size / 2,
                                          center_x + cell_size / 2, center_y + cell_size / 2,
                                          outline=color, width=2, tags='hidden_selection')
            inspected = self.hovered_hidden if self.hovered_hidden in self.hidden_nodes else self.selected_hidden
            detail = f"Ativação: 0 a {max_h_act:.4f}"
            if inspected in self.hidden_nodes:
                detail = (f"H{inspected} [{inspected // columns}, {inspected % columns}]"
                          f" | ativação {hidden_acts[inspected]:.5f}")
            self.create_text(grid_left, grid_top + rows * cell_size + 18, anchor='w', text=detail,
                             font=('Consolas', 10), fill='#f8fafc', tags='hidden_detail')

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
        pix_info = f"Pixel Selecionado: {self.hovered_pixel or self.selected_pixel or '--'}"
        self.create_text(20, height - 16, anchor='w',
                 text=f"{pix_info} | Neurônio oculto: {self.selected_hidden if self.selected_hidden is not None else '--'}",
                         fill='#94a3b8', font=('Bahnschrift', 9))
