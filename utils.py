import csv
import io
import numpy as np

def parse_states(states_text):
    # states_text: string con estados separados por comas
    states = [s.strip() for s in states_text.split(",") if s.strip()]
    if len(states) == 0:
        raise ValueError("Debe especificar al menos un estado.")
    return states

def parse_matrix_from_text(matrix_text):
    # matrix_text: líneas con comas, ejemplo:
    # 0.6,0.3,0.1
    # 0.4,0.4,0.2
    # ...
    f = io.StringIO(matrix_text.strip())
    reader = csv.reader(f)
    mat = []
    for row in reader:
        if len(row) == 0:
            continue
        mat.append([float(x.strip()) for x in row])
    if len(mat) == 0:
        raise ValueError("La matriz está vacía.")
    return np.array(mat, dtype=float)

def read_matrix_csv(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.read()
    return parse_matrix_from_text(lines)
