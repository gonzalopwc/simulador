import numpy as np
import scipy.sparse as sp
from scipy.linalg import logm

class DensityMatrix:
    """
    Representa un estado cuántico mediante una matriz densidad.
    """

    def __init__(self, num_qubits: int, probability_i: np.ndarray = None, psi_i: np.ndarray = None):
        """
        Inicializa el estado cuántico para un número dado de qubits.

        Parámetros:
            num_qubits (int): Número de qubits. Debe ser un entero positivo.
            probability_i (np.ndarray): Probabilidades de los estados mixtos (opcional).
            psi_i (np.ndarray): Vectores de estado correspondientes a cada probabilidad (opcional).
        """
        # Validación de entrada
        if not isinstance(num_qubits, int) or num_qubits <= 0:
            raise ValueError("num_qubits debe ser un entero positivo.")

        self.num_qubits = num_qubits

        # Estado puro por defecto si no se especifica un estado mixto
        if probability_i is None and psi_i is None:
            self.matrix = self._initialize_base_matrix()
        else:
            if len(probability_i) != len(psi_i):
                raise ValueError("probability_i y psi_i deben tener la misma longitud.")
            if len(probability_i) > 2**num_qubits:
                raise ValueError(f"probability_i y psi_i deben tener longitud ≤ {2**num_qubits}.")
            prob_0 = np.sum(probability_i)
            if not (0 <= prob_0 <= 1):
                raise ValueError("La suma de las probabilidades debe estar entre 0 y 1.")
            self.matrix = self._initialize_base_matrix_mixed(probability_i, psi_i, prob_0)

    def _initialize_base_matrix(self) -> np.ndarray:
        """
        Genera la matriz densidad para un estado base puro |0⟩.

        Retorna:
            np.ndarray: Matriz densidad del estado inicial.
        """
        matrix = np.zeros((2**self.num_qubits, 2**self.num_qubits), dtype=complex)
        matrix[0, 0] = 1.0
        return matrix

    def _initialize_base_matrix_mixed(self, probability_i, psi_i, prob_0):
        """
        Genera una matriz densidad para un estado mixto.

        Parámetros:
            probability_i (np.ndarray): Lista de probabilidades.
            psi_i (np.ndarray): Lista de vectores de estado.
            prob_0 (float): Suma de probabilidades.
        
        Retorna:
            np.ndarray: Matriz densidad del estado mixto.
        """
        matrix = np.zeros((2**self.num_qubits, 2**self.num_qubits), dtype=complex)
        matrix[0, 0] = 1.0 - prob_0

        for i in range(len(probability_i)):
            matrix += probability_i[i] * np.outer(psi_i[i], psi_i[i].conj())

        return matrix

    def operate_gate(self, gate: np.ndarray, qubit_target: int) -> None:
        """
        Aplica una compuerta cuántica a un qubit específico del sistema.

        Parámetros:
            gate (np.ndarray): Matriz que representa la compuerta cuántica.
            qubit_target (int): Índice del qubit objetivo (comenzando desde 0).

        Lanza:
            TypeError, ValueError, IndexError: Si la entrada es inválida.
        """
        if not isinstance(gate, np.ndarray):
            raise TypeError("La compuerta debe ser un array de NumPy.")

        if gate.ndim != 2 or gate.shape[0] != gate.shape[1]:
            raise ValueError("La compuerta debe ser una matriz cuadrada 2D.")

        gate_size = gate.shape[0]
        gate_qubits = int(np.log2(gate_size))

        if 2**gate_qubits != gate_size:
            raise ValueError(f"La compuerta de tamaño {gate_size} no es una potencia de 2.")

        if not isinstance(qubit_target, int):
            raise TypeError("El índice del qubit debe ser un entero.")
        if qubit_target < 0 or qubit_target + gate_qubits > self.num_qubits:
            raise IndexError(f"Índice inválido para aplicar una compuerta de {gate_qubits} qubits.")

        # Aplicación directa si el tamaño coincide
        if gate_size == self.matrix.shape[0]:
            self.matrix = np.dot(np.dot(gate, self.matrix), gate.conj().T)
            return

        # Posiciones para producto tensorial
        post_identities = self.num_qubits - gate_qubits - qubit_target

        # Construye la compuerta completa tensorizando con matrices identidad
        full_operator = sp.kron(sp.eye(2**qubit_target), gate, format="coo")
        full_operator = sp.kron(full_operator, sp.eye(2**post_identities), format="coo")

        # Aplica la compuerta
        self.matrix = full_operator @ self.matrix @ full_operator.conj().T

    def observe_qubit_state(self, qubit: int, rand: callable = np.random.rand) -> int:
        """
        Mide un qubit específico y colapsa el estado de acuerdo con el resultado.

        Parámetros:
            qubit (int): Índice del qubit a medir (comienza desde 0).
            rand (callable): Generador aleatorio entre [0,1). Por defecto es np.random.rand.

        Retorna:
            int: Resultado de la medición (0 o 1).

        Lanza:
            TypeError, ValueError: Si el índice del qubit es inválido.
        """
        if not isinstance(qubit, int):
            raise TypeError("El índice del qubit debe ser un entero.")
        if qubit < 0 or qubit >= self.num_qubits:
            raise ValueError(f"Índice del qubit {qubit} fuera de rango.")

        # Proyectores |1⟩ y |0⟩
        P0 = np.array([[1, 0], [0, 0]], dtype=complex)
        P1 = np.array([[0, 0], [0, 1]], dtype=complex)

        # Construye el proyector para |1⟩
        proyector = sp.kron(sp.eye(2**(self.num_qubits - qubit - 1)), P1, format="coo")
        proyector = sp.kron(proyector, sp.eye(2**qubit), format="coo")

        aux = proyector @ self.matrix
        prob_one = np.trace(proyector @ self.matrix)

        # Resultado de la medición
        if prob_one > rand():
            self.matrix = aux @ proyector / prob_one
            return 1
        else:
            # Colapso hacia |0⟩
            proyector = sp.kron(sp.eye(2**(self.num_qubits - qubit - 1)), P0, format="coo")
            proyector = sp.kron(proyector, sp.eye(2**qubit), format="coo")
            self.matrix = proyector @ self.matrix @ proyector / (1 - prob_one)
            return 0

    def von_neumann_entropy(self) -> float:
        """
        Calcula la entropía de Von Neumann de la matriz densidad.

        Retorna:
            float: Valor de la entropía.
        """
        try:
            return -np.trace(self.matrix @ logm(self.matrix)).real
        except np.linalg.LinAlgError:
            return 0.0  # Si la matriz es singular (estado puro)

    def partial_trace (self, remove):
        """
        Calcula la traza parcial eliminando un qubit específico del sistema.

        Parámetros:
            remove (int): Índice del qubit a eliminar (comienza desde 0).

        Retorna:
            np.ndarray: Matriz densidad reducida después de trazar el qubit.
        """

        if remove >= self.num_qubits or remove < 0:
            raise ValueError(f"El índice del qubit debe estar entre 0 y {self.num_qubits - 1}")

        # Obtener la matriz de densidad (|ψ⟩⟨ψ|)
        rho = self.matrix

        # Reshapear a un tensor de 2^n x 2^n en [2, 2, ..., 2] (2 por cada qubit)
        rho_tensor = rho.reshape([2] * (2 * self.num_qubits))

        # Determinar los ejes a trazar (para el qubit 'remove')
        axis1 = remove  # Dimensión del ket del qubit a eliminar
        axis2 = self.num_qubits + remove  # Dimensión del bra del qubit a eliminar

        # Realizar la traza parcial sobre los ejes correspondientes
        rho_reduced = np.trace(rho_tensor, axis1=axis1, axis2=axis2)

        # El resultado ya está en la forma 2^(n-1) x 2^(n-1)
        return rho_reduced
