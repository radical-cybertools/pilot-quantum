# QDREAMER Examples

This directory contains examples demonstrating the integration of QDREAMER (Quantum Resource Allocation and Management Engine) with Pilot Quantum.

## Overview

QDREAMER provides intelligent resource selection for quantum computing tasks, automatically choosing the best quantum backend based on:
- Circuit requirements (qubit count, gate set)
- Resource characteristics (fidelity, error rate, queue length)
- Optimization objectives (high fidelity, high speed, balanced)

## Examples

### 1. Basic QDREAMER Integration (`pq_pcs_ray_dreamer.py`)
- Simple demonstration of QDREAMER with a single pilot
- Shows basic resource selection and task execution

### 2. Multi-Pilot with Fake Backends (`pq_pcs_ray_dreamer_fake_backends.py`)
- Demonstrates multi-pilot architecture with Qiskit local and IBMQ fake backends
- Shows intelligent load balancing across different executor types
- Uses PuLP-based optimization for resource selection

### 3. Framework-Provided Backends (`pq_pcs_ray_dreamer_framework_backends.py`)
- Uses framework-provided backends (Qiskit Aer simulators)
- Demonstrates automatic backend discovery and configuration
- Shows high-performance simulation capabilities

### 4. Custom Backend Configuration (`pq_pcs_ray_dreamer_with_custom_backends.py`)
- Demonstrates custom backend configuration with noise models
- Shows how to define custom quantum resources with specific characteristics
- Includes noise modeling and error rate configuration

### 5. PennyLane Integration (`pq_pcs_ray_dreamer_pennylane.py`)
- Demonstrates QDREAMER with PennyLane quantum circuits
- Shows integration with different PennyLane devices (default.qubit, qiskit.aer)
- Includes hybrid quantum-classical computation examples

## Key Features Demonstrated

### Intelligent Resource Selection
- Automatic selection of best quantum backend based on circuit requirements
- Optimization modes: high fidelity, high speed, balanced
- Real-time queue monitoring and load balancing

### Multi-Pilot Architecture
- Multiple pilots with different executor types
- Cross-pilot resource selection and scheduling
- Distributed task execution with Ray

### Resource Optimization
- PuLP-based linear programming for optimal resource selection
- Consideration of fidelity, queue length, and noise characteristics
- Dynamic resource allocation based on current system state

### Circuit Compatibility
- Automatic gate set compatibility checking
- Qubit count requirements validation
- Backend-specific circuit optimization

## Running the Examples

### Prerequisites
```bash
pip install -r requirements.txt
```

### Basic Example
```bash
python examples/dreamer/pq_pcs_ray_dreamer.py
```

### Multi-Pilot Example
```bash
python examples/dreamer/pq_pcs_ray_dreamer_fake_backends.py
```

### PennyLane Example
```bash
python examples/dreamer/pq_pcs_ray_dreamer_pennylane.py
```

## Configuration

### QDREAMER Configuration
```python
# Initialize QDREAMER with optimization mode
pcs.initialize_dreamer({"optimization_mode": "high_fidelity"})

# Available optimization modes:
# - "high_fidelity": Prioritize high-fidelity backends
# - "high_speed": Prioritize low-queue backends  
# - "balanced": Balance fidelity and speed
```

### Pilot Configuration
```python
# Qiskit Local Pilot
pilot_qiskit = {
    "resource_type": "quantum",
    "quantum": {
        "executor": "qiskit_local",
        "backend": ["qasm_simulator", "statevector_simulator"]
    },
    "type": "ray"
}

# IBMQ Pilot
pilot_ibmq = {
    "resource_type": "quantum", 
    "quantum": {
        "executor": "ibmq",
        "backend": ["fake_quebec", "fake_montreal"]
    },
    "type": "ray"
}
```

## Architecture

### Components
1. **Pilot Compute Service**: Manages pilots and task submission
2. **QDREAMER**: Intelligent resource selection engine
3. **Quantum Executors**: Backend-specific execution engines
4. **Resource Generator**: Creates and manages quantum resources
5. **Worker Processes**: Distributed task execution

### Resource Selection Flow
1. Task submission with circuit requirements
2. QDREAMER analyzes available resources
3. Optimization algorithm selects best resource
4. Task executed on selected backend
5. Results returned with performance metrics

## Performance Monitoring

All examples include comprehensive logging and metrics:
- Task execution time
- Resource selection decisions
- Queue utilization
- Circuit execution results

## Troubleshooting

### Common Issues
1. **Ray Connection Issues**: Ensure Ray cluster is running
2. **Backend Not Found**: Check backend names and availability
3. **Circuit Compatibility**: Verify gate set compatibility
4. **Memory Issues**: Adjust resource allocation for large circuits

### Debug Mode
Enable debug logging by setting the log level:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

When adding new examples:
1. Follow the naming convention: `pq_pcs_ray_dreamer_*.py`
2. Include comprehensive documentation
3. Add error handling and logging
4. Test with different optimization modes
5. Update this README with new features
