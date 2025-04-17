import numpy as np

# ========================================
# Compuertas Cuánticas Fijas
# ========================================

# ----------------------------------------
# Compuertas de Pauli (un solo qubit)
# ----------------------------------------

# Compuerta Pauli-X (equivalente al NOT clásico)
X = np.array([[0, 1],
              [1, 0]], dtype=complex)

# Compuerta Pauli-Y
Y = np.array([[0, -1j],
              [1j, 0]], dtype=complex)

# Compuerta Pauli-Z
Z = np.array([[1, 0],
              [0, -1]], dtype=complex)

# ----------------------------------------
# Compuerta de Hadamard (crea superposición)
# ----------------------------------------

H = (1 / np.sqrt(2)) * np.array([[1, 1],
                                 [1, -1]], dtype=complex)

# ----------------------------------------
# Compuertas de dos qubits
# ----------------------------------------

# Compuerta SWAP (intercambia dos qubits)
SWAP = np.array([[1, 0, 0, 0],
                 [0, 0, 1, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1]], dtype=complex)

# Compuerta CNOT (control en el qubit 0, objetivo en el qubit 1)
CNOT = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0]], dtype=complex)



# ================================================
# Compuertas Cuánticas Parametrizadas (Callables)
# ================================================

# ----------------------------------------
# Matriz Identidad para múltiples qubits
# ----------------------------------------

def I(num_qubits: int) -> np.ndarray:
    """
    Mantiene el estado de entrada
    """
    return np.eye(N=2**num_qubits, M=2**num_qubits, dtype=complex)


# ----------------------------------------
# Compuerta de Rotación RX
# ----------------------------------------

def RX(theta: float) -> np.ndarray:
    """
    Realiza una rotación alrededor del eje X en un ángulo 'theta'.
    """
    return np.array([
        [np.cos(theta / 2), -1j * np.sin(theta / 2)],
        [-1j * np.sin(theta / 2), np.cos(theta / 2)]
    ], dtype=complex)


# ----------------------------------------
# Compuerta de Rotación RY
# ----------------------------------------

def RY(theta: float) -> np.ndarray:
    """
    Realiza una rotación alrededor del eje Y en un ángulo 'theta'.
    """
    return np.array([
        [np.cos(theta / 2), -np.sin(theta / 2)],
        [np.sin(theta / 2),  np.cos(theta / 2)]
    ], dtype=complex)


# ----------------------------------------
# Compuerta de Rotación RZ
# ----------------------------------------

def RZ(theta: float) -> np.ndarray:
    """
    Realiza una rotación alrededor del eje Z en un ángulo 'theta'.
    """
    return np.array([
        [np.exp(-1j * theta / 2), 0],
        [0, np.exp(1j * theta / 2)]
    ], dtype=complex)



