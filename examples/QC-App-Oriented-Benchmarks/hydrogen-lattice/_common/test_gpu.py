from qiskit import  transpile, execute
from qiskit_aer import AerSimulator
from qiskit.circuit.library import QuantumVolume

# Create the AerSimulator with statevector method and GPU device
sim = AerSimulator(method='statevector', device='GPU')

# Define the quantum circuit
qubit = 5  # Number of qubits
circ = transpile(QuantumVolume(qubit, 10, seed=0))  # Transpile the QuantumVolume circuit
circ.measure_all()  # Add measurement to all qubits

# Execute the circuit on the simulator
result = execute(circ, sim, shots=100, blocking_enable=True, blocking_qubits=23).result()

# Access the result
print(result.get_counts())
