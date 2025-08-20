# Multi-Backend Quantum Support in Pilot-Quantum

## Overview

Pilot-Quantum now supports multiple quantum backends including Qiskit, Pennylane, AWS Braket, and IBM Quantum Runtime. This document describes the complete multi-backend architecture and how to use different quantum providers.

## Supported Backends

### 1. **Qiskit Backends**

#### Local Simulators
- **Qiskit Aer**: High-performance simulator with noise modeling
- **Qiskit BasicAer**: Simple simulator for basic operations

```python
quantum_config = {
    "executor": "qiskit_local"
}
```

#### IBM Quantum Runtime
- **IBM Quantum Hardware**: Real quantum computers
- **IBM Quantum Simulators**: Cloud-based simulators

```python
quantum_config = {
    "executor": "ibmq_qasm_simulator",
    "ibmqx_token": "your_ibmq_token"
}
```

### 2. **Pennylane Backends**

#### Default Device
- **Pennylane Default**: Pure Python simulator

```python
quantum_config = {
    "executor": "pennylane_default"
}
```

#### Specialized Devices
- **Pennylane Lightning**: GPU-accelerated simulator
- **Pennylane Qiskit**: Qiskit backend integration
- **Pennylane Braket**: AWS Braket integration
- **Pennylane Forest**: Rigetti Forest integration

```python
quantum_config = {
    "executor": "pennylane_lightning"  # or pennylane_qiskit, pennylane_braket, pennylane_forest
}
```

### 3. **AWS Braket Backends**

#### Amazon Braket Devices
- **SV1**: State vector simulator
- **TN1**: Tensor network simulator
- **IonQ**: IonQ quantum computers
- **Rigetti**: Rigetti quantum computers
- **OQC**: Oxford Quantum Circuits

```python
quantum_config = {
    "executor": "braket_sv1",
    "braket_device_arn": "arn:aws:braket:us-east-1::device/qpu/amazon/sv1"
}
```

## Architecture

### 1. **Resource Discovery**

The `QuantumResourceGenerator` automatically discovers available resources for each executor:

```python
def get_quantum_resources_for_executor(self, executor, config=None):
    if executor == 'qiskit_local':
        return self._get_qiskit_local_resources()
    elif executor.startswith('ibmq_'):
        return self._get_ibmq_resources(executor, config)
    elif executor.startswith('braket_'):
        return self._get_braket_resources(executor, config)
    elif executor.startswith('pennylane_'):
        return self._get_pennylane_resources(executor, config)
```

### 2. **Circuit Execution**

Each backend has specialized execution methods:

```python
def _execute_quantum_circuit(self, quantum_task, resource, *args, **kwargs):
    resource_name = resource.name.lower()
    
    if 'qiskit' in resource_name:
        return self._execute_qiskit_circuit(quantum_task, resource, *args, **kwargs)
    elif 'pennylane' in resource_name:
        return self._execute_pennylane_circuit(quantum_task, resource, *args, **kwargs)
    elif 'braket' in resource_name:
        return self._execute_braket_circuit(quantum_task, resource, *args, **kwargs)
    elif 'ibmq' in resource_name:
        return self._execute_ibmq_circuit(quantum_task, resource, *args, **kwargs)
```

### 3. **QDREAMER Resource Selection**

QDREAMER selects the best resource across all available backends based on:
- Qubit count requirements
- Gate set compatibility
- Queue status and availability
- Error rates and noise levels
- Cost considerations

## Usage Examples

### Example 1: Qiskit with IBM Quantum

```python
from pilot.dreamer import QuantumTask
from pilot.pilot_compute_service import ExecutionEngine, PilotComputeService

def qiskit_circuit():
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc

# Create quantum pilot
pcs = PilotComputeService(execution_engine=ExecutionEngine.RAY, working_directory="/tmp")

quantum_pilot_desc = {
    "resource_type": "quantum",
    "quantum": {
        "executor": "ibmq_qasm_simulator",
        "ibmqx_token": "your_token",
    },
    "working_directory": "/tmp",
    "type": "ray",
    "dreamer_enabled": True,
}

pilot = pcs.create_pilot(pilot_compute_description=quantum_pilot_desc)
pcs.initialize_dreamer()

# Submit quantum task
qt = QuantumTask(
    circuit=qiskit_circuit,
    num_qubits=2,
    gate_set=["h", "cx", "measure"],
    resource_config={"num_qpus": 1, "num_gpus": 0, "memory": None}
)

task = pcs.submit_quantum_task(qt)
pcs.wait_tasks([task])
result = pcs.get_results([task])
```

### Example 2: Pennylane with Multiple Backends

```python
def pennylane_circuit():
    import pennylane as qml
    dev = qml.device("default.qubit", wires=2)
    
    @qml.qnode(dev)
    def circuit():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))
    
    return circuit

# Create quantum pilot with Pennylane Lightning
quantum_pilot_desc = {
    "resource_type": "quantum",
    "quantum": {
        "executor": "pennylane_lightning",
    },
    "working_directory": "/tmp",
    "type": "ray",
    "dreamer_enabled": True,
}

pilot = pcs.create_pilot(pilot_compute_description=quantum_pilot_desc)
pcs.initialize_dreamer()

# Submit quantum task
qt = QuantumTask(
    circuit=pennylane_circuit,
    num_qubits=2,
    gate_set=["h", "cnot"],
    resource_config={"num_qpus": 1, "num_gpus": 0, "memory": None}
)

task = pcs.submit_quantum_task(qt)
pcs.wait_tasks([task])
result = pcs.get_results([task])
```

### Example 3: AWS Braket

```python
def braket_circuit():
    from braket.circuits import Circuit
    circuit = Circuit()
    circuit.h(0)
    circuit.cnot(0, 1)
    circuit.measure([0, 1], [0, 1])
    return circuit

# Create quantum pilot with AWS Braket
quantum_pilot_desc = {
    "resource_type": "quantum",
    "quantum": {
        "executor": "braket_sv1",
        "braket_device_arn": "arn:aws:braket:us-east-1::device/qpu/amazon/sv1",
    },
    "working_directory": "/tmp",
    "type": "ray",
    "dreamer_enabled": True,
}

pilot = pcs.create_pilot(pilot_compute_description=quantum_pilot_desc)
pcs.initialize_dreamer()

# Submit quantum task
qt = QuantumTask(
    circuit=braket_circuit,
    num_qubits=2,
    gate_set=["h", "cnot", "measure"],
    resource_config={"num_qpus": 1, "num_gpus": 0, "memory": None}
)

task = pcs.submit_quantum_task(qt)
pcs.wait_tasks([task])
result = pcs.get_results([task])
```

## Configuration

### Environment Variables

Set the following environment variables for cloud quantum services:

```bash
# IBM Quantum
export IBMQX_TOKEN="your_ibmq_token"

# AWS Braket
export BRAKET_DEVICE_ARN="arn:aws:braket:us-east-1::device/qpu/amazon/sv1"
export AWS_DEFAULT_REGION="us-east-1"
```

### Dependencies

Install required packages:

```bash
pip install qiskit qiskit-ibm-runtime qiskit-aer
pip install pennylane pennylane-lightning pennylane-qiskit pennylane-forest
pip install amazon-braket-sdk
```

## Advanced Features

### 1. **Cross-Backend Scheduling**

QDREAMER can schedule tasks across multiple backends:

```python
# Create multiple quantum pilots with different backends
pilot1 = pcs.create_pilot({
    "resource_type": "quantum",
    "quantum": {"executor": "qiskit_local"},
    "type": "ray",
    "dreamer_enabled": True,
})

pilot2 = pcs.create_pilot({
    "resource_type": "quantum",
    "quantum": {"executor": "pennylane_lightning"},
    "type": "ray",
    "dreamer_enabled": True,
})

# QDREAMER will automatically select the best backend for each task
```

### 2. **Hybrid Quantum-Classical Workflows**

```python
# Classical preprocessing
@ray.remote
def classical_preprocessing(data):
    return processed_data

# Quantum computation
def quantum_circuit(processed_data):
    # Quantum circuit using processed data
    pass

# Classical post-processing
@ray.remote
def classical_postprocessing(quantum_result):
    return final_result

# Execute hybrid workflow
prep_task = classical_preprocessing.remote(data)
quantum_task = pcs.submit_quantum_task(QuantumTask(
    circuit=quantum_circuit,
    num_qubits=4,
    gate_set=["h", "cnot"],
    resource_config={}
))
post_task = classical_postprocessing.remote(quantum_task)

# Wait for all tasks
ray.get([prep_task, quantum_task, post_task])
```

### 3. **Resource Monitoring and Optimization**

```python
# Get cache information
cache_info = pcs.qdreamer.get_cache_info()
print(f"Cache status: {cache_info}")

# Clear cache if needed
pcs.qdreamer.clear_cache()

# Monitor background queue updates
pcs.qdreamer.start_background_monitoring(interval_seconds=30)
```

## Testing

Run comprehensive tests for all backends:

```bash
cd examples
python test_all_quantum_backends.py
```

This will test:
- Qiskit local simulators
- Pennylane default and specialized devices
- IBM Quantum Runtime (if token provided)
- AWS Braket (if device ARN provided)

## Performance Considerations

### 1. **Caching Strategy**
- Queue information cached for 30 seconds (configurable)
- Background monitoring updates every 60 seconds
- Reduces API calls by 25-30x

### 2. **Resource Selection**
- QDREAMER considers current queue status
- Real-time error rates and noise levels
- Cost optimization for cloud services

### 3. **Load Balancing**
- Automatic distribution across available backends
- Queue-aware scheduling
- Failover to alternative backends

## Future Enhancements

### 1. **Additional Backends**
- **Rigetti**: Direct Rigetti Forest integration
- **IonQ**: IonQ quantum computers
- **Google**: Google Quantum AI
- **Microsoft**: Azure Quantum

### 2. **Advanced Features**
- **Circuit Optimization**: Automatic circuit optimization before execution
- **Error Mitigation**: Built-in error mitigation techniques
- **Cost Optimization**: Cost-aware resource selection
- **Multi-Cloud**: Seamless multi-cloud quantum computing

### 3. **Integration**
- **Kubernetes**: Native Kubernetes support
- **Slurm**: Enhanced SLURM integration
- **Cloud Platforms**: AWS, Azure, GCP native integration
