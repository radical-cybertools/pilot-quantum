# Pilot-Quantum

Last Updated: 01/24/2026

# Overview:

Pilot-Quantum is presented as a Quantum-HPC middleware framework designed to address the challenges of integrating quantum and classical computing resources. It focuses on managing heterogeneous resources, including diverse Quantum Processing Unit (QPU) modalities and various integration types with classical resources, such as accelerators.
 
Requirements:

	* Currently only SLURM clusters are supported
	* Setup password-less SSH, e.g., using sshproxy on Perlmutter.

## Installation

### Using uv (Recommended)

Create a virtual environment and install dependencies:

```bash
# Install uv if not already installed
# Create virtual environment
uv venv .venv

# Activate the environment
source .venv/bin/activate  # On Linux/Mac
# or
.venv\Scripts\activate  # On Windows

# Install pilot-quantum in editable mode
uv pip install -e .
```

### Using standard venv and pip

```bash
# Create virtual environment
python3 -m venv .venv

# Activate the environment
source .venv/bin/activate  # On Linux/Mac

# Install pilot-quantum in editable mode
pip install -e .
```

### Development Installation

For development with additional testing dependencies:

```bash
# With uv
uv pip install -e ".[dev]"

# With pip
pip install -e ".[dev]"
```

### Optional: PennyLane Examples

To run the PennyLane examples:

```bash
# With uv
uv pip install -e ".[examples]"

# With pip
pip install -e ".[examples]"
```

## API Usage

Here is a simple script that launches Pythonic functions as tasks on remote SLURM nodes using Pilot-Quantum framework.

```python

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


## Hints

Your default Python environment (activated in `.bashrc` or shell profile) should contain all Pilot-Quantum and application dependencies for remote execution on compute nodes.


## Citation

If you use this work, please cite:

```bibtex
@inproceedings{mantha2025pilotquantum,
  author    = {Mantha, Pradeep and Kiwit, Florian J. and Saurabh, Nishant and Jha, Shantenu and Luckow, Andre},
  title     = {Pilot-Quantum: A Middleware for Quantum-HPC Resource, Workload and Task Management},
  booktitle = {2025 IEEE 25th International Symposium on Cluster, Cloud and Internet Computing (CCGrid)},
  year      = {2025},
  pages     = {1--10},
  doi       = {10.1109/CCGRID64434.2025.00070},
  url       = {https://doi.ieeecomputersociety.org/10.1109/CCGRID64434.2025.00070}
}
```