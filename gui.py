import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox, filedialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import networkx as nx

from markov import validate_transition_matrix, state_distribution_after_n_steps
from utils import parse_states, read_matrix_csv

# Paleta de colores minimalista inspirada en el mockup
COLORS = {
    'primary_blue': '#71A9DB',       # Azul claro principal
    'dark_blue': '#2d4a5e',          # Azul oscuro sidebar
    'accent_orange': '#FFBF6B',      # Naranja/Dorado
    'bg_canvas': '#f5f3ed',          # Fondo beige claro
    'bg_sidebar': '#3a5a72',         # Fondo sidebar
    'bg_dark': '#1e3344',            # Fondo muy oscuro
    'text_light': '#ffffff',         # Texto claro
    'text_dark': '#2b3442',          # Texto oscuro
    'hover_blue': '#5a8bb8',         # Hover azul
}

ctk.set_appearance_mode("Dark")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("MarkovClima")
        self.configure(fg_color=COLORS['bg_dark'])
        self.after(100, lambda: self.state('zoomed'))
        
        # Container principal
        main_container = ctk.CTkFrame(self, fg_color=COLORS['bg_dark'], corner_radius=0)
        main_container.pack(fill="both", expand=True, padx=20, pady=20)
        main_container.grid_rowconfigure(0, weight=1)
        main_container.grid_columnconfigure(1, weight=1)
        
        # ---------- SIDEBAR IZQUIERDO ----------
        self.sidebar = ctk.CTkFrame(
            main_container,
            width=280,
            corner_radius=16,
            fg_color=COLORS['bg_sidebar']
        )
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=(0, 15))
        self.sidebar.grid_propagate(False)
        
        # Header con logo
        header = ctk.CTkFrame(self.sidebar, fg_color="transparent", height=80)
        header.pack(fill="x", padx=20, pady=(25, 20))
        
        ctk.CTkLabel(
            header,
            text="☁️",
            font=ctk.CTkFont(size=32)
        ).pack(side="left", padx=(0, 10))
        
        ctk.CTkLabel(
            header,
            text="MarkovClima",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=COLORS['text_light']
        ).pack(side="left")
        
        # Sección: Estados
        self._create_section_label("Estados")
        self.states_entry = self._create_input("Soleado, Nublado, Lluvioso")
        
        self.generate_btn = ctk.CTkButton(
            self.sidebar,
            text="Generar Matriz",
            command=self.generate_matrix_inputs,
            height=38,
            corner_radius=8,
            fg_color=COLORS['primary_blue'],
            hover_color=COLORS['hover_blue'],
            font=ctk.CTkFont(size=13),
            text_color=COLORS['text_dark']
        )
        self.generate_btn.pack(fill="x", padx=20, pady=(0, 15))
        
        # Sección: Condiciones iniciales
        self._create_section_label("Condiciones Iniciales")
        
        ctk.CTkLabel(
            self.sidebar,
            text="Estado inicial",
            font=ctk.CTkFont(size=12),
            text_color="#b8c5d0",
            anchor="w"
        ).pack(fill="x", padx=20, pady=(0, 5))
        
        self.initial_option = ctk.CTkOptionMenu(
            self.sidebar,
            values=["Soleado", "Nublado", "Lluvioso"],
            height=36,
            corner_radius=8,
            fg_color=COLORS['dark_blue'],
            button_color=COLORS['primary_blue'],
            button_hover_color=COLORS['hover_blue'],
            font=ctk.CTkFont(size=12),
            dropdown_fg_color=COLORS['bg_sidebar']
        )
        self.initial_option.pack(fill="x", padx=20, pady=(0, 12))
        self.initial_option.set("Soleado")
        
        ctk.CTkLabel(
            self.sidebar,
            text="Días a simular",
            font=ctk.CTkFont(size=12),
            text_color="#b8c5d0",
            anchor="w"
        ).pack(fill="x", padx=20, pady=(0, 5))
        
        self.n_spin = self._create_input("3")
        
        ctk.CTkLabel(
            self.sidebar,
            text="Clima objetivo",
            font=ctk.CTkFont(size=12),
            text_color="#b8c5d0",
            anchor="w"
        ).pack(fill="x", padx=20, pady=(0, 5))
        
        self.expected_option = ctk.CTkOptionMenu(
            self.sidebar,
            values=["Soleado", "Nublado", "Lluvioso"],
            height=36,
            corner_radius=8,
            fg_color=COLORS['dark_blue'],
            button_color=COLORS['primary_blue'],
            button_hover_color=COLORS['hover_blue'],
            font=ctk.CTkFont(size=12),
            dropdown_fg_color=COLORS['bg_sidebar']
        )
        self.expected_option.pack(fill="x", padx=20, pady=(0, 20))
        self.expected_option.set("Nublado")
        
        # Botones de acción - en la parte inferior
        actions_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        actions_frame.pack(side="bottom", fill="x", padx=20, pady=20)
        
        self.validate_btn = ctk.CTkButton(
            actions_frame,
            text="Calcular",
            command=self.on_calculate,
            height=44,
            corner_radius=10,
            fg_color=COLORS['accent_orange'],
            hover_color="#e5a855",
            text_color=COLORS['text_dark'],
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.validate_btn.pack(fill="x", pady=(0, 8))
        
        self.reset_btn = ctk.CTkButton(
            actions_frame,
            text="Reiniciar",
            command=self.on_reset,
            height=38,
            corner_radius=10,
            fg_color="transparent",
            hover_color="#2d4a5e",
            font=ctk.CTkFont(size=13),
            border_width=1,
            border_color="#4a6479"
        )
        self.reset_btn.pack(fill="x")
        
        # ---------- ÁREA PRINCIPAL ----------
        self.main_area = ctk.CTkFrame(
            main_container,
            corner_radius=16,
            fg_color=COLORS['bg_canvas']
        )
        self.main_area.grid(row=0, column=1, sticky="nsew")
        self.main_area.grid_rowconfigure(1, weight=1)
        self.main_area.grid_columnconfigure(0, weight=1)
        
        # Top bar
        topbar = ctk.CTkFrame(self.main_area, fg_color="transparent", height=60)
        topbar.grid(row=0, column=0, sticky="ew", padx=25, pady=(20, 0))
        topbar.grid_columnconfigure(0, weight=1)
        
        ctk.CTkLabel(
            topbar,
            text="Matriz de Transición",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color=COLORS['text_dark'],
            anchor="w"
        ).grid(row=0, column=0, sticky="w")
        
        # Contenedor de matriz y resultados
        content_frame = ctk.CTkFrame(self.main_area, fg_color="transparent")
        content_frame.grid(row=1, column=0, sticky="nsew", padx=25, pady=20)
        content_frame.grid_rowconfigure(0, weight=1)
        content_frame.grid_columnconfigure((0, 1), weight=1)
        
        # Panel matriz
        matrix_panel = ctk.CTkFrame(
            content_frame,
            corner_radius=12,
            fg_color="white"
        )
        matrix_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        matrix_panel.grid_rowconfigure(1, weight=1)
        matrix_panel.grid_columnconfigure(0, weight=1)
        
        ctk.CTkLabel(
            matrix_panel,
            text="Valores de Transición",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=COLORS['text_dark'],
            anchor="w"
        ).grid(row=0, column=0, sticky="w", padx=20, pady=(15, 10))
        
                # Crear un canvas con scroll horizontal
        self.matrix_canvas = tk.Canvas(
            matrix_panel,
            bg="white",
            highlightthickness=0
        )
        self.matrix_canvas.grid(row=1, column=0, sticky="nsew", padx=15, pady=(0, 15))

        # Scrollbar horizontal
        x_scroll = tk.Scrollbar(matrix_panel, orient="horizontal", command=self.matrix_canvas.xview)
        x_scroll.grid(row=2, column=0, sticky="ew", padx=15)

        self.matrix_canvas.configure(xscrollcommand=x_scroll.set)

        # Frame dentro del canvas donde se insertarán las entradas
        self.matrix_frame = tk.Frame(self.matrix_canvas, bg="white")
        self.matrix_window = self.matrix_canvas.create_window((0, 0), window=self.matrix_frame, anchor="nw")

        # Actualizar scroll cuando cambie el tamaño del contenido
        def on_frame_configure(event):
            self.matrix_canvas.configure(scrollregion=self.matrix_canvas.bbox("all"))

        self.matrix_frame.bind("<Configure>", on_frame_configure)

        
        # Panel resultados
        results_panel = ctk.CTkFrame(
            content_frame,
            corner_radius=12,
            fg_color="white"
        )
        results_panel.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        results_panel.grid_rowconfigure(1, weight=2)
        results_panel.grid_rowconfigure(2, weight=3)
        results_panel.grid_columnconfigure(0, weight=1)
        
        ctk.CTkLabel(
            results_panel,
            text="Resultados",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=COLORS['text_dark'],
            anchor="w"
        ).grid(row=0, column=0, sticky="w", padx=20, pady=(15, 10))
        
        # Texto de resultados (marco con altura fija + bloqueo de propagación)
        text_container = ctk.CTkFrame(
            results_panel,
            fg_color="#f8f8f8",
            corner_radius=8,
            height=250  # altura fija en píxeles; ajusta a tu gusto
        )
        # Evitar que el frame cambie su tamaño en respuesta al contenido
        text_container.grid(row=1, column=0, sticky="nsew", padx=15, pady=(0, 10))
        text_container.grid_propagate(False)  # <-- clave: no dejar que el contenido cambie el tamaño

        # Asegurarse de que la fila que contiene el text_container no "estire"
        # results_panel.grid_rowconfigure(1, weight=0)  # opcional si ya configuraste antes; puede añadirse

        # Crear un Text con fuente grande, pero sin que el contenedor cambie de tamaño
        self.output_text = tk.Text(
            text_container,
            bg="#f8f8f8",
            fg=COLORS['text_dark'],
            insertbackground=COLORS['accent_orange'],
            font=("Consolas", 14),  # tamaño de letra más grande
            relief="flat",
            padx=12,
            pady=12,
            wrap="word"
        )
        # Colocamos el text dentro del frame fijo y lo expandimos para llenar el espacio interior
        self.output_text.grid(row=0, column=0, sticky="nsew")

        # Opcional: si querés scroll vertical dentro del cuadro sin agrandar el contenedor
        y_scroll = tk.Scrollbar(text_container, orient="vertical", command=self.output_text.yview)
        y_scroll.grid(row=0, column=1, sticky="ns", padx=(0,4), pady=4)
        self.output_text.configure(yscrollcommand=y_scroll.set)

        # Aseguramos que el text ocupe su celda correctamente
        text_container.grid_rowconfigure(0, weight=1)
        text_container.grid_columnconfigure(0, weight=1)

        
        # Gráfico
        graph_container = ctk.CTkFrame(results_panel, fg_color="#f8f8f8", corner_radius=8)
        graph_container.grid(row=2, column=0, sticky="nsew", padx=15, pady=(0, 15))
        
        self.fig = plt.Figure(figsize=(6, 4), dpi=100, facecolor='#f8f8f8')
        self.ax = self.fig.add_subplot(111, facecolor='#f8f8f8')
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_container)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)
        
        # Inicializar
        self.generate_matrix_inputs()


    def _create_section_label(self, text):
        """Crea una etiqueta de sección en el sidebar"""
        ctk.CTkLabel(
            self.sidebar,
            text=text,
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#8a9aa8",
            anchor="w"
        ).pack(fill="x", padx=20, pady=(15, 8))
    
    def _create_input(self, placeholder):
        """Crea un campo de entrada estilizado"""
        entry = ctk.CTkEntry(
            self.sidebar,
            height=36,
            corner_radius=8,
            fg_color=COLORS['dark_blue'],
            border_width=0,
            text_color=COLORS['text_light'],
            font=ctk.CTkFont(size=12)
        )
        entry.pack(fill="x", padx=20, pady=(0, 12))
        entry.insert(0, placeholder)
        return entry

    def generate_matrix_inputs(self):
        for widget in self.matrix_frame.winfo_children():
            widget.destroy()

        states = parse_states(self.states_entry.get())
        self.initial_option.configure(values=states)
        self.expected_option.configure(values=states)
        if states:
            self.initial_option.set(states[0])
            self.expected_option.set(states[-1])

        n = len(states)
        self.matrix_entries = []

        # Encabezado
        tk.Label(
            self.matrix_frame,
            text="",
            bg="white",
            fg=COLORS['text_dark'],
            font=("Arial", 10, "bold")
        ).grid(row=0, column=0, padx=8, pady=8)
        
        for j, col in enumerate(states):
            lbl = tk.Label(
                self.matrix_frame,
                text=col,
                bg=COLORS['primary_blue'],
                fg="white",
                font=("Arial", 10, "bold"),
                padx=12,
                pady=8,
                relief="flat"
            )
            lbl.grid(row=0, column=j+1, padx=3, pady=3, sticky="ew")

        # Filas
        for i, row in enumerate(states):
            lbl = tk.Label(
                self.matrix_frame,
                text=row,
                bg=COLORS['primary_blue'],
                fg="white",
                font=("Arial", 10, "bold"),
                padx=12,
                pady=8,
                relief="flat"
            )
            lbl.grid(row=i+1, column=0, padx=3, pady=3, sticky="ew")
            
            row_entries = []
            for j in range(n):
                e = tk.Entry(
                    self.matrix_frame,
                    width=12,
                    justify="center",
                    bg="#f9f9f9",
                    fg=COLORS['text_dark'],
                    insertbackground=COLORS['accent_orange'],
                    relief="solid",
                    borderwidth=1,
                    font=("Arial", 14)
                )
                e.insert(0, "0.0")
                e.grid(row=i+1, column=j+1, padx=3, pady=3)
                row_entries.append(e)
            self.matrix_entries.append(row_entries)

    def get_matrix_from_inputs(self):
        n = len(self.matrix_entries)
        M = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                val = float(self.matrix_entries[i][j].get())
                M[i, j] = val
        return M

    def load_csv(self):
        path = filedialog.askopenfilename(filetypes=[('CSV files', '*.csv')])
        if not path: return
        try:
            M = read_matrix_csv(path)
            s = ', '.join([f'E{i}' for i in range(1, M.shape[0]+1)])
            self.states_entry.delete(0, 'end')
            self.states_entry.insert(0, s)
            self.generate_matrix_inputs()
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    self.matrix_entries[i][j].delete(0, 'end')
                    self.matrix_entries[i][j].insert(0, str(M[i, j]))
            messagebox.showinfo('✓ Éxito', f'Matriz cargada correctamente')
        except Exception as e:
            messagebox.showerror('Error', str(e))

    def on_reset(self):
        self.states_entry.delete(0, 'end')
        self.states_entry.insert(0, 'Soleado, Nublado, Lluvioso')
        self.n_spin.delete(0, 'end')
        self.n_spin.insert(0, '3')
        self.generate_matrix_inputs()
        self.output_text.delete('1.0', 'end')
        self.ax.clear()
        self.ax.set_facecolor('#f8f8f8')
        self.canvas.draw()

    def on_calculate(self):
        try:
            states = parse_states(self.states_entry.get())
            M = self.get_matrix_from_inputs()
            if M.shape[0] != len(states):
                raise ValueError("Número de estados y dimensiones no coinciden.")
            M = validate_transition_matrix(M)
            n = int(self.n_spin.get())
            init = self.initial_option.get()
            expected = self.expected_option.get()

            if init not in states or expected not in states:
                raise ValueError("Estado no reconocido.")

            init_vector = np.zeros(len(states))
            init_vector[states.index(init)] = 1.0
            dist, Pn = state_distribution_after_n_steps(M, init_vector, n)

            # Resultados
            self.output_text.delete('1.0', 'end')
            dfP = pd.DataFrame(M, index=states, columns=states)
            dfPn = pd.DataFrame(Pn, index=states, columns=states)
            
            self.output_text.insert('end', f"╔══ Matriz P ══╗\n")
            self.output_text.insert('end', f"{dfP.round(4).to_string()}\n\n")
            self.output_text.insert('end', f"╔══ P^{n} (después de {n} días) ══╗\n")
            self.output_text.insert('end', f"{dfPn.round(4).to_string()}\n\n")
            self.output_text.insert('end', f"╔══ Distribución (inicial: {init}) ══╗\n")
            
            for s, p in zip(states, dist):
                line = f"  {s:<12} {p:.4f} ({p*100:5.2f}%)"
                if s == expected:
                    line += "  ← objetivo"
                self.output_text.insert('end', line + "\n")

            prob_expected = dist[states.index(expected)]
            self.output_text.insert('end', f"\n» Probabilidad día {n} = '{expected}': {prob_expected*100:.2f}%\n")

            # ========== GRÁFICO MEJORADO ==========
            self.ax.clear()
            self.ax.set_facecolor('#f8f8f8')
            self.fig.patch.set_facecolor('#f8f8f8')
            
            G = nx.DiGraph()
            
            # Agregar todas las transiciones con peso > 0
            edges_to_draw = []
            for i, a in enumerate(states):
                for j, b in enumerate(states):
                    if M[i, j] > 0.01:  # Filtrar probabilidades muy pequeñas
                        G.add_edge(a, b, weight=M[i, j])
                        edges_to_draw.append((a, b, M[i, j]))
            
            # Layout circular con más espacio
            pos = nx.circular_layout(G, scale=2.5)
            
            # Separar self-loops de aristas normales
            self_loops = [(u, v, w) for u, v, w in edges_to_draw if u == v]
            normal_edges = [(u, v, w) for u, v, w in edges_to_draw if u != v]
            
            # Agrupar aristas por pares bidireccionales
            drawn_pairs = set()
            
            for u, v, weight in normal_edges:
                # Verificar si ya dibujamos este par
                pair = tuple(sorted([u, v]))
                
                # Verificar si hay reversa
                reverse_weight = None
                for a, b, w in normal_edges:
                    if a == v and b == u:
                        reverse_weight = w
                        break
                
                if reverse_weight is not None:
                    # Es bidireccional - dibujar dos flechas con colores diferentes
                    if pair not in drawn_pairs:
                        drawn_pairs.add(pair)
                        
                        # Flecha 1: u -> v (azul)
                        nx.draw_networkx_edges(
                            G, pos,
                            edgelist=[(u, v)],
                            edge_color=COLORS['primary_blue'],
                            arrows=True,
                            arrowsize=20,
                            arrowstyle='-|>',
                            width=2.5,
                            connectionstyle='arc3,rad=0.25',
                            alpha=0.8,
                            node_size=4000,
                            ax=self.ax,
                            min_source_margin=30,
                            min_target_margin=30
                        )
                        
                        # Flecha 2: v -> u (naranja)
                        nx.draw_networkx_edges(
                            G, pos,
                            edgelist=[(v, u)],
                            edge_color=COLORS['accent_orange'],
                            arrows=True,
                            arrowsize=20,
                            arrowstyle='-|>',
                            width=2.5,
                            connectionstyle='arc3,rad=0.25',
                            alpha=0.8,
                            node_size=4000,
                            ax=self.ax,
                            min_source_margin=30,
                            min_target_margin=30
                        )
                else:
                    # Unidireccional - solo una flecha azul
                    nx.draw_networkx_edges(
                        G, pos,
                        edgelist=[(u, v)],
                        edge_color=COLORS['primary_blue'],
                        arrows=True,
                        arrowsize=20,
                        arrowstyle='-|>',
                        width=2.5,
                        connectionstyle='arc3,rad=0.15',
                        alpha=0.8,
                        node_size=4000,
                        ax=self.ax,
                        min_source_margin=30,
                        min_target_margin=30
                    )
            
            # Dibujar self-loops (bucles sobre el mismo nodo)
            for u, v, weight in self_loops:
                from matplotlib.patches import FancyArrowPatch
                node_pos = pos[u]
                
                # Offset para el loop (hacia afuera del círculo)
                angle = np.arctan2(node_pos[1], node_pos[0])
                offset_x = np.cos(angle) * 0.55
                offset_y = np.sin(angle) * 0.55
                
                # Puntos de inicio y fin del loop
                start = (node_pos[0] + offset_x * 0.6, node_pos[1] + offset_y * 0.6)
                end = (node_pos[0] + offset_x * 0.8, node_pos[1] + offset_y * 0.8)
                
                # Crear el arco del self-loop
                loop = FancyArrowPatch(
                    start, end,
                    connectionstyle=f"arc3,rad=1.8",
                    arrowstyle='-|>',
                    mutation_scale=22,
                    linewidth=2.8,
                    color=COLORS['primary_blue'],
                    alpha=0.75,
                    zorder=2
                )
                self.ax.add_patch(loop)
            
            # Dibujar nodos grandes y claros
            nx.draw_networkx_nodes(
                G, pos,
                node_size=4000,
                node_color=COLORS['primary_blue'],
                edgecolors=COLORS['dark_blue'],
                linewidths=3,
                ax=self.ax
            )
            
            # Etiquetas de nodos con mejor contraste
            nx.draw_networkx_labels(
                G, pos,
                font_size=11,
                font_color='white',
                font_weight='bold',
                ax=self.ax
            )
            
            # Etiquetas de probabilidades - con colores que coincidan con las flechas
            drawn_label_pairs = set()
            
            for u, v, weight in normal_edges:
                pos_u = np.array(pos[u])
                pos_v = np.array(pos[v])
                
                # Verificar si hay reversa
                reverse_weight = None
                for a, b, w in normal_edges:
                    if a == v and b == u:
                        reverse_weight = w
                        break
                
                pair = tuple(sorted([u, v]))
                
                if reverse_weight is not None:
                    # Bidireccional - dibujar ambas etiquetas
                    if pair not in drawn_label_pairs:
                        drawn_label_pairs.add(pair)
                        
                        # Calcular posiciones para ambas etiquetas
                        mid = (pos_u + pos_v) / 2
                        direction = pos_v - pos_u
                        perpendicular = np.array([-direction[1], direction[0]])
                        perpendicular = perpendicular / np.linalg.norm(perpendicular)
                        
                        # Etiqueta 1: u -> v (azul) - arriba
                        label_pos1 = mid + perpendicular * 0.35
                        self.ax.text(
                            label_pos1[0], label_pos1[1],
                            f'{weight:.2f}',
                            fontsize=9,
                            fontweight='bold',
                            ha='center',
                            va='center',
                            color='white',
                            bbox=dict(
                                boxstyle='round,pad=0.35',
                                facecolor=COLORS['primary_blue'],
                                edgecolor=COLORS['dark_blue'],
                                linewidth=1.5,
                                alpha=0.9
                            ),
                            zorder=5
                        )
                        
                        # Etiqueta 2: v -> u (naranja) - abajo
                        label_pos2 = mid - perpendicular * 0.35
                        self.ax.text(
                            label_pos2[0], label_pos2[1],
                            f'{reverse_weight:.2f}',
                            fontsize=9,
                            fontweight='bold',
                            ha='center',
                            va='center',
                            color=COLORS['text_dark'],
                            bbox=dict(
                                boxstyle='round,pad=0.35',
                                facecolor=COLORS['accent_orange'],
                                edgecolor='#d49a4a',
                                linewidth=1.5,
                                alpha=0.9
                            ),
                            zorder=5
                        )
                else:
                    # Unidireccional - solo una etiqueta azul
                    mid = (pos_u + pos_v) / 2
                    direction = pos_v - pos_u
                    perpendicular = np.array([-direction[1], direction[0]])
                    perpendicular = perpendicular / np.linalg.norm(perpendicular)
                    label_pos = mid + perpendicular * 0.2
                    
                    self.ax.text(
                        label_pos[0], label_pos[1],
                        f'{weight:.2f}',
                        fontsize=9,
                        fontweight='bold',
                        ha='center',
                        va='center',
                        color='white',
                        bbox=dict(
                            boxstyle='round,pad=0.35',
                            facecolor=COLORS['primary_blue'],
                            edgecolor=COLORS['dark_blue'],
                            linewidth=1.5,
                            alpha=0.9
                        ),
                        zorder=5
                    )
            
            # Etiquetas para self-loops
            for u, v, weight in self_loops:
                node_pos = pos[u]
                angle = np.arctan2(node_pos[1], node_pos[0])
                label_x = node_pos[0] + np.cos(angle) * 0.6
                label_y = node_pos[1] + np.sin(angle) * 0.6
                
                self.ax.text(
                    label_x, label_y,
                    f'{weight:.2f}',
                    fontsize=9,
                    fontweight='bold',
                    ha='center',
                    va='center',
                    color=COLORS['text_dark'],
                    bbox=dict(
                        boxstyle='round,pad=0.4',
                        facecolor='white',
                        edgecolor=COLORS['primary_blue'],
                        linewidth=1.5,
                        alpha=0.95
                    ),
                    zorder=5
                )
            
            self.ax.set_title(
                "Matriz de Transición - Probabilidades entre Estados",
                fontsize=12,
                fontweight='bold',
                color=COLORS['text_dark'],
                pad=20
            )
            
            self.ax.axis('off')
            self.ax.margins(0.15)
            
            # Ajustar límites para que todo se vea bien
            self.ax.set_xlim(-3.2, 3.2)
            self.ax.set_ylim(-3.2, 3.2)
            
            self.canvas.draw()

        except Exception as e:
            messagebox.showerror("Error", str(e))


if __name__ == "__main__":
    app = App()
    app.mainloop()