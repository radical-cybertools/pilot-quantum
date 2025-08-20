# QDREAMER Initialization and Quantum Executors

## Overview

This document describes the QDREAMER initialization process and how to use different quantum executors in Pilot-Quantum. QDREAMER is automatically initialized with quantum resources based on the pilot's quantum configuration.

## Key Features

### 1. Automatic QDREAMER Initialization

QDREAMER is now initialized with quantum resources specific to the executor type specified in the pilot configuration. This allows for:

- **Executor-specific resources**: Different quantum backends for different executors
- **Dynamic resource discovery**: Automatic discovery of available quantum resources
- **Configuration-based setup**: Easy configuration through pilot description

### 2. Supported Quantum Executors

#### Qiskit Local Simulator (`qiskit_local`)
- **Resources**: Qiskit Aer and BasicAer simulators
- **Features**: Perfect simulation with no noise
- **Use case**: Development and testing

```python
quantum_config = {
    "executor": "qiskit_local"
}
```

#### IBM Quantum Backends (`ibmq_*`)
- **Resources**: IBM Quantum Runtime Service backends
- **Features**: Real quantum hardware and simulators
- **Requirements**: IBM Quantum token
- **Use case**: Production quantum computing

```python
quantum_config = {
    "executor": "ibmq_qasm_simulator",
    "ibmqx_token": "your_token_here"
}
```

#### Pennylane Default (`pennylane_default`)
- **Resources**: Pennylane default.qubit device
- **Features**: Hybrid quantum-classical computation
- **Use case**: Quantum machine learning

```python
quantum_config = {
    "executor": "pennylane_default"
}
```

## Usage Pattern

### 1. Create Quantum Pilot

```python
from pilot.pilot_compute_service import ExecutionEngine, PilotComputeService

# Create PilotComputeService
pcs = PilotComputeService(
    execution_engine=ExecutionEngine.RAY, 
    working_directory="/tmp"
)

# Create quantum pilot with specific executor
quantum_pilot_desc = {
    "resource_type": "quantum",
    "quantum": {
        "executor": "qiskit_local",  # or "ibmq_qasm_simulator", "pennylane_default"
        # "ibmqx_token": "your_token"  # required for IBMQ backends
    },
    "working_directory": "/tmp",
    "type": "ray",
    "dreamer_enabled": True,
}

pilot = pcs.create_pilot(pilot_compute_description=quantum_pilot_desc)
```

### 2. Initialize QDREAMER

```python
# Initialize QDREAMER with quantum resources for the executor
pcs.initialize_dreamer()

# Check available resources
print(f"Available quantum resources: {list(pcs.quantum_resources.keys())}")
```

### 3. Submit Quantum Tasks

```python
from pilot.dreamer import QuantumTask

# Create quantum task
qt = QuantumTask(
    circuit=your_circuit_function,
    num_qubits=2,
    gate_set=["h", "cx", "measure"],
    resource_config={"num_qpus": 1, "num_gpus": 0, "memory": None}
)

# Submit task (QDREAMER will select the best resource)
task = pcs.submit_quantum_task(qt)

# Wait and get results
pcs.wait_tasks([task])
result = pcs.get_results([task])
```

## Implementation Details

### Quantum Resource Discovery

The `QuantumResourceGenerator` class now supports executor-specific resource discovery:

```python
def get_quantum_resources_for_executor(self, executor, config=None):
    if executor == 'qiskit_local':
        return self._get_qiskit_local_resources()
    elif executor.startswith('ibmq_'):
        return self._get_ibmq_resources(executor, config)
    elif executor == 'pennylane_default':
        return self._get_pennylane_resources()
    else:
        return self._get_fake_backend_resources()
```

### QDREAMER Configuration

QDREAMER is automatically configured with:

- **Queue dynamics**: Default queue management for each resource
- **Resource mapping**: Mapping of resource names to QuantumResource objects
- **Simulation mode**: Enabled for testing and development

### Resource Selection

QDREAMER selects the best quantum resource based on:

1. **Qubit count compatibility**: Resource must have enough qubits
2. **Gate set compatibility**: Resource must support required gates
3. **Queue status**: Resource queue must not be full
4. **Error rates**: Lower error rates are preferred
5. **Noise levels**: Lower noise levels are preferred

## Examples

### Example 1: Qiskit Local Simulator

```python
# Pilot configuration
quantum_pilot_desc = {
    "resource_type": "quantum",
    "quantum": {
        "executor": "qiskit_local",
    },
    "working_directory": "/tmp",
    "type": "ray",
    "dreamer_enabled": True,
}

# Create pilot and initialize QDREAMER
pilot = pcs.create_pilot(pilot_compute_description=quantum_pilot_desc)
pcs.initialize_dreamer()

# Available resources: ['qiskit_aer_simulator', 'qiskit_basicaer_simulator']
```

### Example 2: IBM Quantum Backend

```python
# Pilot configuration
quantum_pilot_desc = {
    "resource_type": "quantum",
    "quantum": {
        "executor": "ibmq_qasm_simulator",
        "ibmqx_token": "your_ibmq_token",
    },
    "working_directory": "/tmp",
    "type": "ray",
    "dreamer_enabled": True,
}

# Create pilot and initialize QDREAMER
pilot = pcs.create_pilot(pilot_compute_description=quantum_pilot_desc)
pcs.initialize_dreamer()

# Available resources: ['ibmq_qasm_simulator'] or actual IBMQ backends
```

### Example 3: Pennylane Default

```python
# Pilot configuration
quantum_pilot_desc = {
    "resource_type": "quantum",
    "quantum": {
        "executor": "pennylane_default",
    },
    "working_directory": "/tmp",
    "type": "ray",
    "dreamer_enabled": True,
}

# Create pilot and initialize QDREAMER
pilot = pcs.create_pilot(pilot_compute_description=quantum_pilot_desc)
pcs.initialize_dreamer()

# Available resources: ['pennylane_default_qubit']
```

## Testing

A comprehensive test script is provided at `examples/test_quantum_executors.py` that demonstrates:

- Qiskit local simulator usage
- Pennylane default device usage
- IBM Quantum backend usage (requires token)
- QDREAMER initialization for each executor type
- Resource discovery and selection

Run the tests with:
```bash
cd examples
python test_quantum_executors.py
```

## Error Handling

### Common Issues and Solutions

1. **QDREAMER not initialized**
   ```
   Error: QDREAMER not initialized. Call initialize_dreamer() after creating a quantum pilot.
   ```
   **Solution**: Call `pcs.initialize_dreamer()` after creating the quantum pilot.

2. **No quantum resources found**
   ```
   Error: No quantum resources found for executor: <executor_name>
   ```
   **Solution**: Check that the executor name is correct and supported.

3. **IBMQ token required**
   ```
   Error: IBMQX_TOKEN not found in environment
   ```
   **Solution**: Set the `IBMQX_TOKEN` environment variable or provide it in the quantum config.

4. **Import errors for quantum providers**
   ```
   Error: IBMQ provider not available
   ```
   **Solution**: Install required quantum packages:
   ```bash
   pip install qiskit-ibm-runtime  # for IBMQ
   pip install pennylane          # for Pennylane
   ```

## Future Enhancements

- **Additional executors**: Support for more quantum providers (Rigetti, IonQ, etc.)
- **Dynamic resource discovery**: Real-time discovery of available quantum resources
- **Advanced resource selection**: More sophisticated QDREAMER algorithms
- **Resource monitoring**: Real-time monitoring of quantum resource status
- **Multi-executor support**: Support for using multiple executors simultaneously
