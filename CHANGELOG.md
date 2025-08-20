# Changelog

All notable changes to Pilot Quantum will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **QDREAMER Integration**: Intelligent quantum resource selection engine
  - PuLP-based optimization for resource selection
  - Real-time queue monitoring and load balancing
  - Multi-pilot resource management
  - Automatic circuit compatibility checking
  - Task correlation IDs for distributed logging

- **New Executor Architecture**: Refactored executor system
  - `BaseExecutor` abstract base class for all executors
  - `QiskitExecutor` for local Qiskit Aer simulators
  - `IBMQExecutor` for IBM Quantum Runtime service
  - `PennyLaneExecutor` for PennyLane quantum circuits
  - `BraketExecutor` for AWS Braket quantum service

- **Quantum Resource Management**:
  - `QuantumResource` class with fidelity calculation
  - `QuantumResourceGenerator` for automatic resource discovery
  - Support for custom backend configurations
  - Framework-provided backend detection

- **Multi-Pilot Architecture**:
  - Support for multiple pilots with different executor types
  - Cross-pilot resource selection and scheduling
  - Intelligent load balancing across pilots
  - Pilot-specific resource allocation

- **Enhanced Task Management**:
  - `QuantumTask` class for quantum circuit execution
  - Circuit metadata validation (qubit count, gate set)
  - Resource requirement specification
  - Task correlation and logging

- **Optimization Modes**:
  - `high_fidelity`: Prioritize high-fidelity backends
  - `high_speed`: Prioritize low-queue backends
  - `balanced`: Balance fidelity and speed

- **Comprehensive Examples**:
  - Basic QDREAMER integration
  - Multi-pilot with fake backends
  - Framework-provided backends
  - Custom backend configuration
  - PennyLane integration

- **Test Suite**:
  - QDREAMER integration tests
  - Performance and memory usage tests
  - Executor compatibility tests
  - Resource selection optimization tests

### Changed
- **Executor Factory**: Simplified executor creation and management
  - Unified executor interface with `BaseExecutor`
  - Streamlined executor registration and discovery
  - Improved error handling and validation

- **Pilot Compute Service**: Enhanced quantum task submission
  - `submit_quantum_task()` method for quantum-specific tasks
  - QDREAMER integration for intelligent resource selection
  - Distributed resource selection at worker level
  - Background queue monitoring

- **PennyLane Integration**: Simplified architecture
  - Single `pennylane` executor (removed `pennylane_default`, `pennylane_qiskit`)
  - Device configuration in pilot description
  - Direct device name specification

- **Resource Configuration**: Improved backend handling
  - List-based backend specification
  - Automatic fallback to available backends
  - Enhanced logging for backend discovery

### Removed
- **Legacy Executor Classes**: Removed old executor implementations
- **Complex Configuration**: Simplified QDREAMER configuration
- **Redundant Parameters**: Removed unused configuration options
- **Temporary Files**: Cleaned up development artifacts

### Fixed
- **Import Issues**: Fixed `BaseQuantumExecutor` → `BaseExecutor` imports
- **Constructor Parameters**: Fixed executor initialization with proper parameters
- **Resource Allocation**: Fixed pilot creation with missing configuration keys
- **Backend Discovery**: Fixed backend name mismatches and fallback logic
- **Task Execution**: Fixed circuit execution with proper executor configuration

## [Previous Versions]

### [0.1.0] - 2024-10-09
- Initial release of Pilot Quantum
- Basic quantum computing integration
- Support for Qiskit and PennyLane
- SLURM cluster integration
- Ray and Dask execution engines

---

## Migration Guide

### For Existing Users

#### Executor Configuration Changes
```python
# Old way
"quantum": {
    "executor": "pennylane_default",
    "backend": "default.qubit"
}

# New way
"quantum": {
    "executor": "pennylane",
    "device": "default.qubit"
}
```

#### Task Submission Changes
```python
# Old way
task_id = pcs.submit_task(quantum_circuit, *args)

# New way
qt = QuantumTask(circuit=quantum_circuit, num_qubits=2, gate_set=["h", "cx"])
task_id = pcs.submit_quantum_task(qt)
```

#### QDREAMER Integration
```python
# Initialize QDREAMER
pcs.initialize_dreamer({"optimization_mode": "high_fidelity"})

# Submit tasks with intelligent resource selection
task_id = pcs.submit_quantum_task(qt)
```

### Breaking Changes
- Executor class names changed (e.g., `BaseQuantumExecutor` → `BaseExecutor`)
- PennyLane executor configuration simplified
- Task submission API enhanced for quantum-specific features
- Resource discovery and allocation improved

### New Features
- QDREAMER intelligent resource selection
- Multi-pilot architecture
- Enhanced logging and monitoring
- Comprehensive test suite
- Performance optimization

---

## Contributing

When contributing to this project, please:
1. Update this changelog with your changes
2. Follow the existing code style
3. Add tests for new functionality
4. Update documentation as needed
