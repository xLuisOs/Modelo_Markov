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

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("MarkovClima - Simulador de Cadenas de Markov")
        self.geometry("1150x700")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        
        # ---------- PANEL IZQUIERDO ----------
        self.controls_frame = ctk.CTkFrame(self, width=400, corner_radius=8)
        self.controls_frame.grid(row=0, column=0, padx=12, pady=12, sticky="ns")
        self.controls_frame.grid_rowconfigure(12, weight=1)
        
        ctk.CTkLabel(self.controls_frame, text="Configuración", font=ctk.CTkFont(size=18, weight="bold")).grid(row=0, column=0, pady=(8,12))

        # Estados
        ctk.CTkLabel(self.controls_frame, text="Estados (separados por comas):").grid(row=1, column=0, sticky="w", padx=8)
        self.states_entry = ctk.CTkEntry(self.controls_frame)
        self.states_entry.grid(row=2, column=0, padx=8, pady=(4,8), sticky="we")
        self.states_entry.insert(0, "Soleado, Nublado, Lluvioso")

        self.generate_btn = ctk.CTkButton(self.controls_frame, text="Generar matriz", command=self.generate_matrix_inputs)
        self.generate_btn.grid(row=3, column=0, padx=8, pady=(0,8), sticky="we")

        # Contenedor de matriz dinámica
        self.matrix_frame = ctk.CTkScrollableFrame(self.controls_frame, height=250)
        self.matrix_frame.grid(row=4, column=0, padx=8, pady=(4,8), sticky="we")

        # Botones de CSV
        self.load_btn = ctk.CTkButton(self.controls_frame, text="Cargar CSV de matriz", command=self.load_csv)
        self.load_btn.grid(row=5, column=0, padx=8, pady=(0,6), sticky="we")

        # Estado inicial
        ctk.CTkLabel(self.controls_frame, text="Estado inicial:").grid(row=6, column=0, sticky="w", padx=8)
        self.initial_option = ctk.CTkOptionMenu(self.controls_frame, values=["Soleado", "Nublado", "Lluvioso"])
        self.initial_option.grid(row=7, column=0, padx=8, pady=(4,8), sticky="we")
        self.initial_option.set("Soleado")

        # Número de días
        ctk.CTkLabel(self.controls_frame, text="Número de días:").grid(row=8, column=0, sticky="w", padx=8)
        self.n_spin = ctk.CTkEntry(self.controls_frame)
        self.n_spin.grid(row=9, column=0, padx=8, pady=(4,8), sticky="we")
        self.n_spin.insert(0, "3")

        # Clima esperado
        ctk.CTkLabel(self.controls_frame, text="Clima esperado en el día n:").grid(row=10, column=0, sticky="w", padx=8)
        self.expected_option = ctk.CTkOptionMenu(self.controls_frame, values=["Soleado", "Nublado", "Lluvioso"])
        self.expected_option.grid(row=11, column=0, padx=8, pady=(4,8), sticky="we")
        self.expected_option.set("Nublado")

        # Botones finales
        btn_frame = ctk.CTkFrame(self.controls_frame, fg_color="transparent")
        btn_frame.grid(row=12, column=0, padx=8, pady=8, sticky="we")
        self.validate_btn = ctk.CTkButton(btn_frame, text="Validar y Calcular", command=self.on_calculate)
        self.validate_btn.grid(row=0, column=0, padx=4, pady=4, sticky="we")
        self.reset_btn = ctk.CTkButton(btn_frame, text="Reiniciar", fg_color="#a3a3a3", command=self.on_reset)
        self.reset_btn.grid(row=0, column=1, padx=4, pady=4, sticky="we")

        # ---------- PANEL DERECHO ----------
        self.results_frame = ctk.CTkFrame(self, corner_radius=8)
        self.results_frame.grid(row=0, column=1, padx=12, pady=12, sticky="nsew")
        self.results_frame.grid_rowconfigure(2, weight=1)
        self.results_frame.grid_columnconfigure(0, weight=1)

        self.title_lbl = ctk.CTkLabel(self.results_frame, text="Resultados", font=ctk.CTkFont(size=18, weight="bold"))
        self.title_lbl.grid(row=0, column=0, sticky="w", padx=12, pady=(8,4))

        self.output_text = tk.Text(self.results_frame, height=12, bg="#1b1b1b", fg="white", insertbackground="white")
        self.output_text.grid(row=1, column=0, padx=12, pady=(4,8), sticky="nsew")

        self.fig = plt.Figure(figsize=(5,3), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.results_frame)
        self.canvas.get_tk_widget().grid(row=2, column=0, padx=12, pady=8, sticky="nsew")

        # Inicial
        self.generate_matrix_inputs()

    # ---------- FUNCIONES ----------
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

        # Etiquetas de columnas
        tk.Label(self.matrix_frame, text="", bg="#2b2b2b", fg="white").grid(row=0, column=0)
        for j, col in enumerate(states):
            tk.Label(self.matrix_frame, text=col, bg="#2b2b2b", fg="white", width=10).grid(row=0, column=j+1, padx=2, pady=2)

        # Filas con entradas
        for i, row in enumerate(states):
            tk.Label(self.matrix_frame, text=row, bg="#2b2b2b", fg="white", width=10).grid(row=i+1, column=0, padx=2, pady=2)
            row_entries = []
            for j in range(n):
                e = tk.Entry(self.matrix_frame, width=8, justify="center")
                e.insert(0, "0.0")
                e.grid(row=i+1, column=j+1, padx=2, pady=2)
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
            messagebox.showinfo('Matriz cargada', f'Matriz cargada desde: {path}')
        except Exception as e:
            messagebox.showerror('Error al cargar CSV', str(e))

    def on_reset(self):
        self.states_entry.delete(0, 'end')
        self.states_entry.insert(0, 'Soleado, Nublado, Lluvioso')
        self.n_spin.delete(0, 'end')
        self.n_spin.insert(0, '3')
        self.generate_matrix_inputs()
        self.output_text.delete('1.0', 'end')
        self.ax.clear()
        self.canvas.draw()

    def on_calculate(self):
        try:
            states = parse_states(self.states_entry.get())
            M = self.get_matrix_from_inputs()
            if M.shape[0] != len(states):
                raise ValueError("Número de estados y dimensiones de la matriz no coinciden.")
            M = validate_transition_matrix(M)
            n = int(self.n_spin.get())
            init = self.initial_option.get()
            expected = self.expected_option.get()

            if init not in states or expected not in states:
                raise ValueError("Estado inicial o esperado no reconocido.")

            init_vector = np.zeros(len(states))
            init_vector[states.index(init)] = 1.0
            dist, Pn = state_distribution_after_n_steps(M, init_vector, n)

            # Mostrar resultados
            self.output_text.delete('1.0', 'end')
            dfP = pd.DataFrame(M, index=states, columns=states)
            dfPn = pd.DataFrame(Pn, index=states, columns=states)
            self.output_text.insert('end', f"Matriz P:\n{dfP.round(4).to_string()}\n\n")
            self.output_text.insert('end', f"P^{n} (después de {n} días):\n{dfPn.round(4).to_string()}\n\n")
            self.output_text.insert('end', f"Distribución después de {n} días (estado inicial: {init}):\n")
            for s, p in zip(states, dist):
                line = f"  {s}: {p:.4f} ({p*100:.2f}%)"
                if s == expected:
                    line += "  ⬅️ Clima esperado"
                self.output_text.insert('end', line + "\n")

            prob_expected = dist[states.index(expected)]
            self.output_text.insert('end', f"\nProbabilidad de que el día {n} esté '{expected}': {prob_expected:.4f} ({prob_expected*100:.2f}%)\n")

            # Graficar el autómata
            self.ax.clear()
            G = nx.DiGraph()
            for i, a in enumerate(states):
                for j, b in enumerate(states):
                    if M[i, j] > 0:
                        G.add_edge(a, b, weight=M[i, j])
            pos = nx.circular_layout(G)
            nx.draw(G, pos, with_labels=True, ax=self.ax, node_size=1200, node_color="#87CEEB", font_color="black", arrows=True)
            edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=self.ax)
            self.ax.set_title("Diagrama de transiciones climáticas")
            self.canvas.draw()

        except Exception as e:
            messagebox.showerror("Error", str(e))
