# Pilot-Quantum

Last Updated: 10/09/2024

# Overview:

Pilot-Quantum is presented as a Quantum-HPC middleware framework designed to address the challenges of integrating quantum and classical computing resources. It focuses on managing heterogeneous resources, including diverse Quantum Processing Unit (QPU) modalities and various integration types with classical resources, such as accelerators.
 
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


## Hints

Your default conda environment should contain all Pilot-Quantum and application dependencies. Activate it, e.g., in the `.bashrc`
