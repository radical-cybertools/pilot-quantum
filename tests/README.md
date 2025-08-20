# Pilot Quantum Test Suite

This directory contains comprehensive tests for the Pilot Quantum framework, ensuring that all components work correctly and efficiently.

## Test Structure

### Test Categories

1. **Basic Tests** (`test_basic.py`)
   - Import validation
   - Component initialization
   - Configuration validation
   - Basic functionality verification

2. **Executor Tests** (`test_all_executors.py`)
   - Qiskit executor functionality
   - PennyLane executor functionality
   - Multi-executor configurations
   - Resource selection and load balancing
   - Error handling and validation

3. **QDREAMER Integration Tests** (`test_qdreamer_integration.py`)
   - QDREAMER initialization and configuration
   - Resource selection and optimization
   - Multi-pilot resource management
   - Task execution with intelligent resource selection
   - Performance characteristics and memory usage

4. **Performance Tests** (`test_all_executors.py`)
   - Execution time measurements
   - Resource usage monitoring
   - Performance threshold validation

### Supported Executors

- `qiskit_local` - Qiskit local simulator
- `qiskit` - Qiskit framework (alias for qiskit_local)
- `ibmq` - IBM Quantum backends (real or fake)
- `pennylane` - PennyLane framework with configurable devices
- `braket_local` - AWS Braket local simulator

### PennyLane Architecture

The PennyLane executor uses a simplified architecture:

- **Single Executor**: `pennylane` (instead of separate `pennylane_default`, `pennylane_qiskit`)
- **Device Configuration**: Specified in `pilot_compute_description`:
  ```python
  "quantum": {
      "executor": "pennylane",
      "device": "default.qubit"  # or "qiskit.aer", "lightning.qubit", etc.
  }
  ```
- **No Backend Parameter**: Device information is read directly from the configuration
- **Raw Circuit Functions**: Circuits return raw functions, devices are configured by the executor

## Running Tests

### Using the Test Runner

The test runner provides a convenient way to run different test categories:

```bash
# Run all tests
python tests/test_runner.py --all

# Run specific test categories
python tests/test_runner.py --basic
python tests/test_runner.py --executors
python tests/test_runner.py --qdreamer
python tests/test_runner.py --performance

# Run a specific test
python tests/test_runner.py --test TestBasicImports.test_pilot_imports

# Generate a test report
python tests/test_runner.py --all --report
```

### Using unittest directly

```bash
# Run all tests
python -m unittest discover tests

# Run specific test file
python -m unittest tests.test_basic

# Run specific test class
python -m unittest tests.test_basic.TestBasicImports

# Run specific test method
python -m unittest tests.test_basic.TestBasicImports.test_pilot_imports
```

## Test Configuration

Test configuration is defined in `test_config.py`:

- **Execution Settings**: Timeouts, retries, wait times
- **Resource Settings**: Qubit limits, fidelity requirements
- **Circuit Settings**: Gate sets, depth limits
- **Executor Configurations**: Device settings for each executor
- **Performance Thresholds**: Execution time and resource usage limits

## Test Environment

### Requirements

- Python 3.8+
- Qiskit
- PennyLane
- Ray
- All Pilot Quantum dependencies

### Environment Setup

Tests use temporary directories for working files and cleanup automatically. The test environment is configured in `test_config.py`:

```python
ENVIRONMENT_CONFIG = {
    'working_directory': '/tmp/pilot_quantum_tests',
    'log_level': 'INFO',
    'cleanup_on_exit': True,
    'save_results': True,
    'results_directory': '/tmp/pilot_quantum_results',
}
```

## Test Results

### Success Criteria

- All tests pass without failures or errors
- Performance tests meet defined thresholds
- Resource selection works correctly
- Error handling functions as expected

### Reporting

Test reports include:
- Test execution summary
- Success/failure counts
- Performance metrics
- Detailed error information
- Execution time statistics

## Continuous Integration

The test suite is designed to run in CI/CD environments:

- Automated test discovery
- Configurable timeouts
- Clean environment setup
- Comprehensive error reporting
- Performance benchmarking

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Timeout Errors**: Increase timeout values in test configuration
3. **Resource Errors**: Check that required quantum backends are available
4. **Performance Failures**: Adjust performance thresholds or investigate performance regressions

### Debug Mode

Run tests with increased verbosity for debugging:

```bash
python -m unittest tests.test_basic -v
```

### Logging

Tests use the standard Python logging framework. Set log level in test configuration or environment variables:

```bash
export PILOT_LOG_LEVEL=DEBUG
python tests/test_runner.py --all
```
