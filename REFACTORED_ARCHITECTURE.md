# Refactored Pilot-Quantum Architecture

## Overview

The Pilot-Quantum framework has been refactored to achieve cleaner separation of concerns, improved modularity, and better extensibility. This document describes the new architecture and how it addresses the previous issues with scattered helper functions and tight coupling.

## Architecture Principles

### 1. Separation of Concerns
- **Execution Logic**: Backend-specific execution logic is isolated in dedicated executor classes
- **Resource Management**: Quantum resource discovery and management is centralized
- **Service Orchestration**: High-level service classes coordinate between components
- **Task Management**: Pilot-Quantum core focuses on task submission and lifecycle management

### 2. Factory Pattern
- **Executor Factory**: Centralized creation and management of quantum executors
- **Resource Factory**: Standardized quantum resource generation across backends
- **Configuration Management**: Consistent configuration handling across components

### 3. Plugin Architecture
- **Extensible Executors**: New quantum backends can be added by implementing `BaseQuantumExecutor`
- **Service Layer**: Business logic is separated from infrastructure concerns
- **Dependency Injection**: Components receive their dependencies rather than creating them

## New Package Structure

```
pilot/
├── executors/                    # Quantum backend execution logic
│   ├── __init__.py
│   ├── base_executor.py         # Abstract base class for all executors
│   ├── qiskit_executor.py       # Qiskit local simulator execution
│   ├── pennylane_executor.py    # Pennylane device execution
│   ├── braket_executor.py       # AWS Braket execution
│   ├── ibmq_executor.py         # IBM Quantum Runtime execution
│   └── executor_factory.py      # Factory for creating executors
├── services/                     # Service layer for orchestration
│   ├── __init__.py
│   └── quantum_execution_service.py  # Main quantum execution service
├── util/
│   └── quantum_resource_generator.py  # Resource discovery and management
├── dreamer.py                   # QDREAMER resource selection
└── pilot_compute_service.py     # Core pilot management (simplified)
```

## Key Components

### 1. BaseQuantumExecutor (pilot/executors/base_executor.py)

**Purpose**: Abstract base class that defines the interface for all quantum executors.

**Key Methods**:
- `execute_circuit(circuit, *args, **kwargs)`: Execute a quantum circuit
- `get_backend_info()`: Return backend capabilities and status
- `is_available()`: Check if the backend is available

**Benefits**:
- Enforces consistent interface across all backends
- Enables polymorphic execution
- Simplifies testing and mocking

### 2. Quantum Executors

#### QiskitExecutor (pilot/executors/qiskit_executor.py)
- Handles Qiskit Aer simulators
- Supports local quantum circuit execution
- Configurable backend selection
- **Compatible with both old and new Qiskit versions** (handles Aer import from qiskit vs qiskit_aer)

#### PennylaneExecutor (pilot/executors/pennylane_executor.py)
- Manages Pennylane devices (default.qubit, Lightning, etc.)
- Supports hybrid quantum-classical computation
- Handles QNode creation and execution

#### BraketExecutor (pilot/executors/braket_executor.py)
- Interfaces with AWS Braket services
- Manages AWS device selection and authentication
- Handles Braket-specific circuit execution

#### IBMQExecutor (pilot/executors/ibmq_executor.py)
- Connects to IBM Quantum Runtime
- Manages IBM Quantum authentication and job submission
- Handles runtime-specific optimizations

### 3. QuantumExecutorFactory (pilot/executors/executor_factory.py)

**Purpose**: Centralized factory for creating and managing quantum executors.

**Key Features**:
- Type normalization (e.g., "qiskit" → QiskitExecutor)
- Configuration validation
- Executor caching
- Support discovery

**Usage**:
```python
executor = QuantumExecutorFactory.create_executor("qiskit", config)
result = executor.execute_circuit(circuit)
```

### 4. QuantumExecutionService (pilot/services/quantum_execution_service.py)

**Purpose**: High-level service that orchestrates quantum circuit execution.

**Key Responsibilities**:
- Resource-to-executor mapping
- Executor lifecycle management
- Error handling and retry logic
- Result formatting and validation

**Benefits**:
- Single point of entry for quantum execution
- Consistent error handling
- Simplified integration with Pilot-Quantum core

### 5. QuantumResourceGenerator (pilot/util/quantum_resource_generator.py)

**Purpose**: Discovers and manages quantum resources across different backends.

**Key Features**:
- Backend-specific resource discovery
- Dynamic resource generation
- Configuration-driven resource creation
- Support for multiple quantum providers

## Refactoring Benefits

### 1. Eliminated Code Duplication
**Before**: Helper functions scattered throughout `pilot_compute_service.py`
```python
def _execute_qiskit_circuit(self, circuit, backend_name):
    # Qiskit-specific logic

def _execute_pennylane_circuit(self, circuit, device_name):
    # Pennylane-specific logic

def _execute_braket_circuit(self, circuit, device_arn):
    # Braket-specific logic
```

**After**: Clean executor classes with single responsibility
```python
class QiskitExecutor(BaseQuantumExecutor):
    def execute_circuit(self, circuit, *args, **kwargs):
        # All Qiskit logic in one place

class PennylaneExecutor(BaseQuantumExecutor):
    def execute_circuit(self, circuit, *args, **kwargs):
        # All Pennylane logic in one place
```

### 2. Improved Testability
- Each executor can be tested independently
- Mock executors for unit testing
- Isolated integration tests per backend

### 3. Enhanced Extensibility
- New backends require only implementing `BaseQuantumExecutor`
- No changes to core Pilot-Quantum logic
- Configuration-driven backend selection

### 4. Better Error Handling
- Backend-specific error handling in executors
- Consistent error reporting through service layer
- Graceful degradation when backends are unavailable

## Migration Guide

### For Existing Code

1. **Update Imports**:
```python
# Old
from pilot.pilot_compute_service import PilotComputeService

# New
from pilot.pilot_compute_service import PilotComputeService
from pilot.services.quantum_execution_service import QuantumExecutionService
```

2. **Quantum Task Submission** (No changes needed):
```python
# This remains the same
pcs.submit_quantum_task(quantum_task)
```

3. **Resource Configuration**:
```python
# Old
pilot_quantum_description = {
    "quantum": {
        "backend": "qiskit_aer",
        "backend_name": "qasm_simulator"
    }
}

# New
pilot_quantum_description = {
    "quantum": {
        "executor": "qiskit",
        "backend_name": "qasm_simulator"
    }
}
```

### For New Backend Integration

1. **Create Executor**:
```python
from pilot.executors.base_executor import BaseQuantumExecutor

class MyBackendExecutor(BaseQuantumExecutor):
    def execute_circuit(self, circuit, *args, **kwargs):
        # Implementation specific to your backend
        pass
    
    def get_backend_info(self):
        return {"name": "my_backend", "qubits": 5}
    
    def is_available(self):
        return True
```

2. **Register with Factory**:
```python
# In executor_factory.py
_executors = {
    'qiskit': QiskitExecutor,
    'pennylane': PennylaneExecutor,
    'braket': BraketExecutor,
    'ibmq': IBMQExecutor,
    'my_backend': MyBackendExecutor,  # Add this line
}
```

3. **Add Resource Generation**:
```python
# In quantum_resource_generator.py
def _get_my_backend_resources(self, executor=None, config=None):
    # Return list of QuantumResource objects for your backend
    pass
```

## Performance Improvements

### 1. Caching
- Executor instances are cached by the service layer
- Resource information is cached with TTL
- QDREAMER queue information is cached and updated in background
- **QDREAMER resource selection happens on main process** using cached instance with background monitoring

### 2. Lazy Loading
- Executors are created only when needed
- Backend connections are established on first use
- Resources are discovered dynamically

### 3. Parallel Execution
- Multiple executors can run concurrently
- Background monitoring doesn't block task execution
- Resource discovery happens in parallel
- **Resource selection and circuit execution are separated**: QDREAMER selects resources on main process, execution happens on remote workers

## Configuration Examples

### Qiskit Local
```python
pilot_quantum_description = {
    "quantum": {
        "executor": "qiskit",
        "backend_name": "qasm_simulator",
        "shots": 1000
    }
}
```

### Pennylane with Lightning
```python
pilot_quantum_description = {
    "quantum": {
        "executor": "pennylane",
        "device": "lightning.qubit",
        "wires": 4
    }
}
```

### AWS Braket
```python
pilot_quantum_description = {
    "quantum": {
        "executor": "braket",
        "device_arn": "arn:aws:braket:us-east-1::device/qpu/ionq/ionQdevice",
        "s3_bucket": "my-braket-bucket"
    }
}
```

### IBM Quantum Runtime
```python
pilot_quantum_description = {
    "quantum": {
        "executor": "ibmq",
        "backend": "ibmq_qasm_simulator",
        "service": "runtime"
    }
}
```

## Testing Strategy

### Unit Tests
- Test each executor independently
- Mock backend dependencies
- Test factory creation and configuration

### Integration Tests
- Test service layer orchestration
- Test resource discovery and mapping
- Test end-to-end quantum task execution

### Performance Tests
- Test caching effectiveness
- Test concurrent executor usage
- Test resource selection performance

## Future Enhancements

### 1. Plugin System
- Dynamic executor loading
- Configuration-driven plugin discovery
- Hot-swappable backends

### 2. Advanced Caching
- Circuit result caching
- Backend state caching
- Distributed caching across nodes

### 3. Monitoring and Metrics
- Executor performance metrics
- Backend availability monitoring
- Resource utilization tracking

### 4. Circuit Optimization
- Backend-specific circuit transpilation
- Gate set optimization
- Noise-aware compilation

## Conclusion

The refactored architecture provides a solid foundation for Pilot-Quantum's continued growth and evolution. The clean separation of concerns, extensible design, and improved maintainability make it easier to add new quantum backends, implement advanced features, and maintain the codebase over time.

The new architecture successfully addresses the previous issues with scattered helper functions by:
- Centralizing backend-specific logic in dedicated executor classes
- Providing a clean service layer for orchestration
- Implementing a factory pattern for resource management
- Enabling easy extension and testing

This refactoring positions Pilot-Quantum as a robust, scalable framework for quantum computing resource management and task execution.
