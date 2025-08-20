# Pilot-Quantum

Last Updated: 08/19/2025

# Overview:

Pilot-Quantum is a Quantum-HPC middleware framework designed to address the challenges of integrating quantum and classical computing resources. It focuses on managing heterogeneous resources, including diverse Quantum Processing Unit (QPU) modalities and various integration types with classical resources, such as accelerators.

## Key Features

- **Multi-Executor Support**: Qiskit, IBMQ, PennyLane, and AWS Braket executors
- **Intelligent Resource Selection**: QDREAMER integration for optimal quantum backend selection
- **Distributed Computing**: Ray and Dask support for scalable quantum computing
- **Multi-Pilot Architecture**: Manage multiple quantum resources simultaneously
- **Real-time Optimization**: PuLP-based resource optimization with queue monitoring
- **Circuit Compatibility**: Automatic gate set and qubit count validation
 
Requirements:

	* Currently only SLURM clusters are supported
	* Setup password-less documentation, e.g., using sshproxy on Perlmutter.

Anaconda or Miniconda is the preferred distribution


## Installation

Create environment with tool of your choice:

    conda create -n pilot-quantum python=3.12

Requirement (in case a manual installation is required):

The best way to utilize Pilot-Quantum is Anaconda, which provides an easy way to install

    pip install -r requirements.txt

To install Pilot-Quantum type:

    python setup.py install

## API Usage

Here is a simple script that launches Pythonic functions as tasks on remote SLURM nodes using Pilot-Quantum framework.

```

from pilot.pilot_compute_service import ExecutionEngine, PilotComputeService

pilot_compute_description = {
    "resource": "slurm://localhost",
    "working_directory": WORKING_DIRECTORY,
    "number_of_nodes": 2,
    "cores_per_node": 1,
    "queue": "premium",
    "walltime": 30,
    "type": "ray",
    "project": "sample",
    "scheduler_script_commands": ["#SBATCH --constraint=cpu"]    
}

def pennylane_quantum_circuit():
    # pennylane circuit definition...
    pass
    
# Pilot-Creation
pcs = PilotComputeService(execution_engine=ExecutionEngine.RAY, working_directory=WORKING_DIRECTORY)
pcs.create_pilot(pilot_compute_description=pilot_compute_description_ray)

# Task submission
tasks = []
for i in range(10):
    k = pcs.submit_task(pennylane_quantum_circuit, i, resources={'num_cpus': 1, 'num_gpus': 0, 'memory': None})
    tasks.append(k)

# Wait for tasks to complete
pcs.wait_tasks(tasks)

# Terminate the pilot
pcs.cancel()
```

## QDREAMER Integration

Pilot-Quantum now includes QDREAMER (Quantum Resource Allocation and Management Engine) for intelligent resource selection:

```python
from pilot.pilot_compute_service import ExecutionEngine, PilotComputeService
from pilot.dreamer import QuantumTask

# Create quantum task
qt = QuantumTask(
    circuit=my_quantum_circuit,
    num_qubits=2,
    gate_set=["h", "cx"],
    resource_config={"num_qpus": 1}
)

# Initialize QDREAMER
pcs.initialize_dreamer({"optimization_mode": "high_fidelity"})

# Submit task with intelligent resource selection
task_id = pcs.submit_quantum_task(qt)
result = pcs.get_results([task_id])
```

### QDREAMER Examples

See `examples/dreamer/` for comprehensive examples:
- Multi-pilot resource management
- Custom backend configuration
- PennyLane integration
- Framework-provided backends
- Intelligent load balancing

## Architecture

### Components
- **Pilot Compute Service**: Core orchestration and pilot management
- **QDREAMER**: Intelligent quantum resource selection engine
- **Quantum Executors**: Backend-specific execution engines
- **Resource Generator**: Quantum resource discovery and management
- **Worker Processes**: Distributed task execution

### Supported Executors
- **QiskitExecutor**: Local Qiskit Aer simulators
- **IBMQExecutor**: IBM Quantum Runtime service
- **PennyLaneExecutor**: PennyLane quantum circuits
- **BraketExecutor**: AWS Braket quantum service

## Hints

Your default conda environment should contain all Pilot-Quantum and application dependencies. Activate it, e.g., in the `.bashrc`
