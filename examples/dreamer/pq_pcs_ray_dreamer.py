import math
import os

import pennylane as qml
import ray
from pilot.dreamer import Coupling, QuantumTask
from pilot.pilot_compute_service import ExecutionEngine, PilotComputeService
from time import sleep
from qiskit import QuantumCircuit

RESOURCE_URL_HPC = "ssh://localhost"
WORKING_DIRECTORY = os.path.join(os.environ["HOME"], "work")

pilot_compute_description_ray = {
    "resource": RESOURCE_URL_HPC,
    "working_directory": WORKING_DIRECTORY,
    "type": "ray",
    "number_of_nodes": 2,
    "cores_per_node": 1,
}

# Example 1: Qiskit local simulator
pilot_quantum_description_local = {
    "resource_type": "quantum",
    "quantum": {
        "executor": "qiskit_local",
    },
    "working_directory": WORKING_DIRECTORY,
    "type": "ray",
    "dreamer_enabled": True,
}

# Example 2: IBM Quantum backend (requires token)
pilot_quantum_description_ibmq = {
    "resource_type": "quantum",
    "quantum": {
        "executor": "ibmq_qasm_simulator",
        "ibmqx_token": "<token>",  # required if not in config file
    },
    "working_directory": WORKING_DIRECTORY,
    "type": "ray",
    "dreamer_enabled": True,
}

# Use local simulator for this example
pilot_quantum_description = pilot_quantum_description_local

qdreamer_config = {}

def start_pilot():
    pcs = PilotComputeService(execution_engine=ExecutionEngine.RAY, 
                              working_directory=WORKING_DIRECTORY)
    
    
    # pcs.create_pilot(pilot_compute_description=pilot_compute_description_ray)
    pcs.create_pilot(pilot_compute_description=pilot_quantum_description)
    return pcs


def qiskit_circuit():    
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])  
    return qc

if __name__ == "__main__":
    pcs = None
    try:
        # Start Pilot
        pcs = start_pilot()

        pcs.initialize_dreamer()
        
        tasks = []
        for i in range(10):
            # Create quantum task with circuit function
            qt = QuantumTask(
                circuit=qiskit_circuit,
                num_qubits=10,  # Match the circuit size
                gate_set=[],  # Gates used in the circuit
                resource_config={"num_qpus": 2, "num_gpus": 0, "memory": None}
            )
            # Submit quantum task - this will automatically route through QDREAMER
            k = pcs.submit_quantum_task(qt)
            tasks.append(k)                    
        pcs.wait_tasks(tasks)
        print(pcs.get_results(tasks))              
    finally:
        if pcs:
            pcs.cancel()