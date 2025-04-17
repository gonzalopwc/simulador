
import numpy as np
import scipy.sparse as sp
from multiprocessing import Pool, cpu_count
from scipy.linalg import logm

class QuantumRegister:
    """
    Clase que representa un registro cuántico utilizando un vector de estado (ket).
    Permite operaciones como aplicar compuertas, mediciones, colapsos, y trazas parciales.
    """

    def __init__(self, num_qubits: int, N_processes: int = None):
        """
        Inicializa un estado cuántico para un número dado de qubits.

        Parámetros:
            num_qubits (int): Número de qubits. Debe ser mayor que cero.
            N_processes (int): Número de procesos en paralelo para operaciones. Por defecto se usa el total de CPUs.
        """

        if not isinstance(num_qubits, int) or num_qubits <= 0:
            raise ValueError("num_qubits debe ser un entero positivo.")

        self.num_qubits = num_qubits
        self.state = self._create_initial_state()

        # Establece el número de procesos
        self.N_processes = cpu_count() if N_processes is None else N_processes

        if not isinstance(self.N_processes, int):
            raise TypeError(f"El número de procesos debe ser un entero, no {type(self.N_processes)}")
        if self.N_processes < 0 or self.N_processes > cpu_count():
            raise ValueError(f"Número de procesos {self.N_processes} fuera de rango. Máximo: {cpu_count()}")


    def _create_initial_state(self) -> np.ndarray:
        """
        Genera el vector de estado inicial |0⟩.

        Devuelve:
            np.ndarray: Vector de estado columna (2^n, 1)
        """

        ket = np.zeros((2**self.num_qubits, 1), dtype=complex)
        ket[0, 0] = 1
        return ket

    def operate_gate(self, gate: np.ndarray, qubit_target: int) -> None:
        """
        Aplica una compuerta cuántica a uno o más qubits específicos del estado cuántico.
        Expande automáticamente la compuerta para que actúe sobre todo el sistema 
        mediante el producto tensorial con matrices identidad.

        Argumentos:
            gate (ndarray): Una compuerta cuántica representada como matriz.
            qubit_target (int): Índice del qubit objetivo menos significativo (base 0).
                Para compuertas de múltiples qubits, este índice especifica dónde empieza
                la acción de la compuerta.

        Excepciones:
            TypeError: Si `gate` no es una matriz válida.
            ValueError: Si las dimensiones de `gate` son incompatibles.
            IndexError: Si la posición del qubit objetivo es inválida.

        Complejidad:
            Espacio O(n) = 2^2n y Ω(n) = 2^n, Tiempo Θ(n) = 2^2n, donde n es el número de qubits.
            Para compuertas de uno o dos qubits (como se espera), se tiene complejidad 2^n.
            Solo compuertas densas completas (como matrices unitarias aleatorias) alcanzan la complejidad 2^2n.
        """
        # Validación de entrada

        if not isinstance(gate, np.ndarray):
            raise TypeError("La compuerta debe ser un arreglo de numpy")

        if gate.ndim != 2 or gate.shape[0] != gate.shape[1]:
            raise ValueError("La compuerta debe ser una matriz cuadrada 2D")

        # Calcular el número de qubits que actúa la compuerta a partir de su tamaño
        gate_size = gate.shape[0]
        gate_qubits = int(np.log2(gate_size))

        if 2**gate_qubits != gate_size:
            raise ValueError(f"El tamaño de la compuerta {gate_size} no es una potencia de 2")

        # Validación del índice del qubit objetivo
        if not isinstance(qubit_target, int):
            raise TypeError(f"El qubit objetivo debe ser un entero, se recibió {type(qubit_target)}")

        if qubit_target < 0 or qubit_target + gate_qubits > self.num_qubits:
            raise IndexError(
                f"Posición de destino {qubit_target} inválida para una compuerta de {gate_qubits} qubits "
                f"en un registro de {self.num_qubits} qubits"
            )

        # Aplicación directa si la compuerta ya es del tamaño del estado completo
        if gate_size == self.state.shape[0]:
            self.state = np.dot(gate, self.state)
            return

        # Cálculo de las posiciones del producto tensorial
        post_identities = self.num_qubits - gate_qubits - qubit_target

        # Construcción del operador completo mediante el producto tensorial con identidades
        full_operator = sp.kron(sp.eye(2**post_identities), gate, format="coo")
        full_operator = sp.kron(full_operator, sp.eye(2**qubit_target), format="coo")

        # Aplicar la compuerta al vector de estado
        self.state = full_operator.dot(self.state)
        return
        
    def calculate_density_matrix(self) -> np.ndarray:
        """Devuelve la matriz de densidad del estado actual"""
        return np.dot(self.state, self.state.conj().T)

    def _calculate_chunk_probability(self, start_end_mask):
        """
        Calcula la probabilidad total de los estados dentro de un rango [start, end),
        considerando solo aquellos índices que cumplen con un cierto patrón de bits (máscara).

        Argumentos:
            start_end_mask (tuple): Una tupla (start, end, mask) donde:
                - start (int): índice inicial del rango.
                - end (int): índice final del rango (no inclusivo).
                - mask (int): máscara de bits; solo se consideran los índices donde (i & mask) != 0.

        Devuelve:
            float: La suma de las probabilidades (|amplitud|²) de los estados seleccionados.
        """
        start, end, mask = start_end_mask
        prob = 0.0

        for i in range(start, end):
            if i & mask:
                prob += np.abs(self.state[i])**2

        return prob

    def _calculate_qubit_probability(self, qubit: int) -> float:
        """
        NO COLAPSA el estado. Calcula la probabilidad de medir el qubit especificado en el estado |1⟩.

        Argumentos:
            qubit (int): Índice del qubit que se desea medir.

        Devuelve:
            float: Probabilidad (0 ≤ p ≤ 1) de medir el estado |1⟩ en el qubit objetivo.

        Excepciones:
            TypeError: Si el índice del qubit no es un entero.
            ValueError: Si el índice del qubit está fuera del rango válido.

        Complejidad:
            Espacio Θ(n, k) = k, Tiempo Θ(n, k) = (2^n)/k
            donde n es el número de qubits y k el número de procesos paralelos.
        """
        # Validación del índice del qubit
        if not isinstance(qubit, int):
            raise TypeError(f"El índice del qubit debe ser un entero, se recibió {type(qubit)}")

        if qubit < 0 or qubit >= self.num_qubits:
            raise ValueError(f"Índice de qubit {qubit} fuera de los límites para un sistema de {self.num_qubits} qubits")

        # Construcción de la máscara de bits para identificar los estados donde el qubit está en |1⟩
        mask = 1 << qubit

        # División del vector de estado en chunks para procesamiento paralelo
        index = np.linspace(0, len(self.state), endpoint=True, num=self.N_processes + 1, dtype=int)
        chunks = [(index[i], index[i + 1], mask) for i in range(self.N_processes)]

        # Procesamiento paralelo para calcular probabilidades por bloques
        pool = Pool(processes=self.N_processes)
        with pool:
            partial = pool.map(self._calculate_chunk_probability, chunks)

        # Suma de probabilidades parciales
        prob_one = sum(partial)

        # Asegurar que el valor esté dentro del rango [0, 1]
        return min(float(prob_one), 1.0)


    def _apply_chunk_collapse(self, start_end_mask_value_prob_one):
        """
        Aplica el colapso del estado cuántico en un rango del vector de estado.

        Argumentos:
            start_end_mask_value_prob_one (tuple): Una tupla con los siguientes elementos:
                - start (int): Índice inicial del bloque.
                - end (int): Índice final del bloque (no inclusivo).
                - mask (int): Máscara para determinar si el qubit está en estado |1⟩.
                - value (int): Resultado de la medición (0 o 1).
                - prob_one (float): Probabilidad de que el qubit esté en estado |1⟩.

        Efecto:
            Colapsa el vector de estado según el resultado de la medición,
            eliminando los estados incompatibles y renormalizando los restantes.
        """
        start, end, mask, value, prob_one = start_end_mask_value_prob_one

        if value == 1:  # Qubit colapsó al estado |1⟩
            for i in range(start, end):
                if not i & mask:
                    self.state[i] = 0  # Eliminar estados incompatibles
                else:
                    self.state[i] /= np.sqrt(prob_one)  # Renormalizar

        else:  # Qubit colapsó al estado |0⟩
            for i in range(start, end):
                if i & mask:
                    self.state[i] = 0  # Eliminar estados incompatibles
                else:
                    self.state[i] /= np.sqrt(1 - prob_one)  # Renormalizar

        return


    def _apply_qubit_collapse(self, qubit: int, value: int, prob_one: float) -> None:
        """
        Colapsa el qubit especificado a un estado clásico (|0⟩ o |1⟩) y renormaliza el vector de estado.

        Argumentos:
            qubit (int): Índice del qubit a colapsar (comenzando desde el bit menos significativo).
            value (int): Valor clásico al que colapsar (0 para |0⟩, 1 para |1⟩).
            prob_one (float): Probabilidad de que el qubit esté en |1⟩ antes del colapso,
                            puede obtenerse con QReg._calculate_qubit_probability(qubit).

        Excepciones:
            TypeError: Si el índice del qubit o el valor no son enteros.
            ValueError: Si el índice está fuera del rango válido.
            ValueError: Si el valor de colapso no es 0 ni 1.
            ValueError: Si la probabilidad no está entre 0 y 1.

        Complejidad:
            Espacio Θ(n,k) = k, Tiempo Θ(n,k) = (2^n) / k
            donde n es el número de qubits y k el número de procesos paralelos.
        """
        # Validación de entradas
        if not isinstance(qubit, int):
            raise TypeError(f"El índice del qubit debe ser un entero, se recibió {type(qubit)}")
        if not isinstance(value, int):
            raise TypeError(f"El valor de colapso debe ser un entero (0 o 1), se recibió {type(value)}")
        if qubit < 0 or qubit >= self.num_qubits:
            raise ValueError(f"Índice de qubit {qubit} fuera de rango para un sistema de {self.num_qubits} qubits")
        if value not in {0, 1}:
            raise ValueError(f"El valor de colapso debe ser 0 o 1, se recibió {value}")
        if not 0 <= prob_one <= 1:
            raise ValueError(f"La probabilidad debe estar entre 0 y 1, se recibió {prob_one}")

        # Máscara para identificar los estados compatibles con el resultado de medición
        mask = 1 << qubit

        # Colapso en paralelo
        if self.N_processes > 1:
            index = np.linspace(0, len(self.state), endpoint=True, num=self.N_processes + 1, dtype=int)
            chunks = [(index[i], index[i + 1], mask, value, prob_one) for i in range(self.N_processes)]

            pool = Pool(processes=self.N_processes)
            with pool:
                pool.map(self._apply_chunk_collapse, chunks)

        else:
            # Colapso secuencial (sin multiprocessing)
            self._apply_chunk_collapse((0, len(self.state), mask, value, prob_one))


    def measure(self, qubit: int, rand: callable = np.random.rand) -> int:
        """
        Mide el qubit especificado, colapsando el sistema cuántico de acuerdo al resultado.

        Argumentos:
            qubit (int): Índice del qubit a medir (empezando desde 0).
            rand (callable): Generador de números aleatorios. Por defecto, np.random.rand.
                            Debe devolver números flotantes en el rango [0, 1).

        Devuelve:
            int: Resultado de la medición (0 o 1).

        Excepciones:
            TypeError: Si el índice del qubit no es un entero.
            ValueError: Si el índice del qubit está fuera de los límites.

        Complejidad:
            Espacio Θ(n) = 1, Tiempo Θ(n) = 2^n, donde n es el número de qubits.
        """
        # Validación del índice del qubit
        if not isinstance(qubit, int):
            raise TypeError(f"El índice del qubit debe ser un entero, se recibió {type(qubit)}")
        if qubit < 0 or qubit >= self.num_qubits:
            raise ValueError(f"Índice de qubit {qubit} fuera de los límites para un sistema de {self.num_qubits} qubits")

        # Calcular la probabilidad de medir |1⟩
        prob_one = self._calculate_qubit_probability(qubit=qubit)

        # Simular la medición comparando con un número aleatorio
        if prob_one > rand():
            self._apply_qubit_collapse(qubit=qubit, value=1, prob_one=prob_one)
            return 1
        else:
            self._apply_qubit_collapse(qubit=qubit, value=0, prob_one=prob_one)
            return 0

    def calculate_density_matrix(self) -> np.ndarray:
        """
        Calcula la matriz de densidad (operador de estado cuántico) del estado cuántico actual.

        La matriz de densidad ρ se define como el producto externo del vector de estado consigo mismo:
            ρ = |ψ⟩⟨ψ|
        donde |ψ⟩ es el vector de estado actual y ⟨ψ| es su transpuesta conjugada compleja.

        Devuelve:
            np.ndarray: Un arreglo 2D de forma (2^n, 2^n) que representa la matriz de densidad,
            donde n es el número de qubits.

        Complejidad:
            Espacio Θ(n) = 2^(2n), Tiempo Θ(n) = 2^(2n), donde n es el número de qubits.
        """
        return np.outer(self.state, self.state.conj())

    def bloch_angles(self):
        """
        Calcula los ángulos de la esfera de Bloch (θ, ϕ) para un estado cuántico de un solo qubit.

        El estado se representa de la siguiente forma:
            ∣ψ⟩ = cos(θ/2)|0⟩ + e^(iϕ)sin(θ/2)|1⟩

        Donde:
            - θ (theta) es el ángulo polar (0 ≤ θ ≤ π)
            - ϕ (phi) es el ángulo azimutal (0 ≤ ϕ < 2π)

        Devuelve:
            tuple[float, float]: Un par de ángulos (theta, phi) en radianes que representan:
                - theta: Ángulo polar desde el eje +Z (rango [0, π])
                - phi: Ángulo azimutal desde el eje +X en el plano X-Y (rango [0, 2π])

        Excepciones:
            ValueError: Si el registro cuántico contiene más de 1 qubit.

        Complejidad:
            Espacio Θ(n) = 1, Tiempo Θ(n) = 1, donde n es el número de qubits.
        """

        # Validar que el registro cuántico contenga solo un qubit
        if self.num_qubits != 1:
            raise ValueError("La esfera de Bloch solo se aplica a un qubit.")

        # Extraer amplitudes alpha y beta del estado
        alpha = self.state[0, 0]  # Amplitud de |0⟩
        beta = self.state[1, 0]   # Amplitud de |1⟩

        # Casos especiales: |0⟩ o |1⟩
        if np.isclose(abs(alpha), 1.0, atol=1e-8):
            return 0.0, 0.0  # Estado |0⟩
        if np.isclose(abs(beta), 1.0, atol=1e-8):
            return np.pi, 0.0  # Estado |1⟩

        # Calcular theta (ángulo polar)
        theta = 2 * np.arccos(abs(alpha))

        if theta == 0 or theta == np.pi:
            return theta, 0.0  # phi=0 por convención

        # Calcular phi (ángulo azimutal) como diferencia de fases
        phi = np.angle(beta) - np.angle(alpha)
        phi = phi % (2 * np.pi)  # Asegurar que phi esté en [0, 2π)

        return theta, phi

    def von_neumann_entropy(self) -> float:
        """
        Devuelve la entropía de Von Neumann de la matriz de densidad.
        La entropía de Von Neumann mide la "mezcla" de un estado cuántico.
        """
        M = self.calculate_density_matrix()  # Obtiene la matriz de densidad
        return -np.trace(M @ logm(M)).real  # Calcula la traza de la matriz de densidad multiplicada por su logaritmo
    
    def trace_out_qubit(self, remove: int) -> np.ndarray:
        """
        Calcula la traza parcial de un sistema de qubits eliminando un qubit especificado.
        
        Parámetros:
            remove: Índice del qubit a eliminar (basado en 0).

        Devuelve:
            np.ndarray: La matriz de densidad reducida después de trazar el qubit especificado.
        """
        
        if remove >= self.num_qubits or remove < 0:
            raise ValueError(f"El índice del qubit debe estar entre 0 y {self.num_qubits - 1}")

        # Obtener la matriz de densidad (|ψ⟩⟨ψ|)
        rho = np.outer(self.state, self.state.conj())

        # Reshapear a un tensor de 2^n x 2^n en [2, 2, ..., 2] (2 por cada qubit)
        rho_tensor = rho.reshape([2] * (2 * self.num_qubits))

        # Determinar los ejes a trazar (para el qubit 'remove')
        axis1 = remove  # Dimensión del ket del qubit a eliminar
        axis2 = self.num_qubits + remove  # Dimensión del bra del qubit a eliminar

        # Realizar la traza parcial sobre los ejes correspondientes
        rho_reduced = np.trace(rho_tensor, axis1=axis1, axis2=axis2)

        # El resultado ya está en la forma 2^(n-1) x 2^(n-1)
        return rho_reduced