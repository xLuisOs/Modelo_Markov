import numpy as np

def validate_transition_matrix(P, tol=1e-8):
    P = np.array(P, dtype=float)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("La matriz P debe ser cuadrada (n x n).")
    col_sums = P.sum(axis=0)
    if not np.allclose(col_sums, np.ones(P.shape[1]), atol=tol):
        raise ValueError(f"Cada columna de P debe sumar 1. Sumas actuales: {col_sums}")
    if np.any(P < -tol) or np.any(P > 1+tol):
        raise ValueError("Elementos de P deben estar en el intervalo [0,1].")
    return P

def matrix_power(P, n):
    P = np.array(P, dtype=float)
    if n < 0:
        raise ValueError("n debe ser entero no negativo")
    return np.linalg.matrix_power(P, int(n))

def state_distribution_after_n_steps(P, initial_vector, n):
    Pn = matrix_power(P, n)
    v = np.array(initial_vector, dtype=float)
    if v.ndim == 1:
        v = v.reshape(-1, 1)
    if v.shape[0] != P.shape[0]:
        raise ValueError("El vector inicial debe tener la misma longitud que el número de estados.")
    result = Pn @ v
    return result.flatten(), Pn
