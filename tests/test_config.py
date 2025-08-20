#!/usr/bin/env python3
"""
Configuration file for Pilot Quantum framework tests.

This file contains test settings, parameters, and configurations
for different test scenarios.
"""

import os

# Test configuration for Pilot Quantum framework

# Execution settings
EXECUTION_CONFIG = {
    'timeout': 300,  # 5 minutes
    'max_retries': 3,
    'wait_time': 10,  # seconds
}

# Resource settings
RESOURCE_CONFIG = {
    'min_qubits': 2,
    'max_qubits': 32,
    'min_fidelity': 0.8,
    'max_error_rate': 0.2,
}

# Circuit settings
CIRCUIT_CONFIG = {
    'max_depth': 10,
    'max_gates': 50,
    'supported_gates': ['h', 'cx', 'x', 'z', 'measure', 'cnot', 'rx', 'ry', 'rz'],
}

# Executor configurations
EXECUTOR_CONFIGS = {
    'qiskit_local': {
        'backend': 'qasm_simulator',
        'shots': 1000,
        'error_rate': 0.0,
        'noise_level': 0.0,
    },
    'qiskit': {
        'backend': 'qasm_simulator',
        'shots': 1000,
        'error_rate': 0.0,
        'noise_level': 0.0,
    },
    'ibmq': {
        'backend': 'fake_ibm_quebec',
        'shots': 1000,
        'error_rate': 0.001,
        'noise_level': 0.002,
    },
    'pennylane': {
        'device': 'default.qubit',
        'wires': 30,
        'shots': 1000,
        'error_rate': 0.0,
        'noise_level': 0.0,
    },
    'braket_local': {
        'device': 'braket.aws.qubit',
        'shots': 1000,
        'error_rate': 0.001,
        'noise_level': 0.002,
    },
}

# Standard test circuits
STANDARD_CIRCUITS = {
    'bell_state': {
        'name': 'Bell State',
        'qubits': 2,
        'gates': ['h', 'cx'],
        'depth': 2,
        'expected_result': 'entangled_state',
    },
    'ghz_state': {
        'name': 'GHZ State',
        'qubits': 3,
        'gates': ['h', 'cx'],
        'depth': 3,
        'expected_result': 'entangled_state',
    },
    'simple_circuit': {
        'name': 'Simple Circuit',
        'qubits': 2,
        'gates': ['x', 'cx'],
        'depth': 2,
        'expected_result': 'product_state',
    },
}

# Performance thresholds
PERFORMANCE_THRESHOLDS = {
    'max_execution_time': 60.0,  # seconds
    'min_execution_time': 0.1,   # seconds
    'max_memory_usage': 1024,    # MB
    'max_cpu_usage': 80.0,       # percentage
}

# Test environment configuration
ENVIRONMENT_CONFIG = {
    'working_directory': '/tmp/pilot_quantum_tests',
    'log_level': 'INFO',
    'cleanup_on_exit': True,
    'save_results': True,
    'results_directory': '/tmp/pilot_quantum_results',
}

# Complete test configuration
TEST_CONFIG = {
    'execution': EXECUTION_CONFIG,
    'resource': RESOURCE_CONFIG,
    'circuit': CIRCUIT_CONFIG,
    'executors': EXECUTOR_CONFIGS,
    'circuits': STANDARD_CIRCUITS,
    'performance': PERFORMANCE_THRESHOLDS,
    'environment': ENVIRONMENT_CONFIG,
}

# Environment-specific settings
ENVIRONMENT_CONFIG = {
    'development': {
        'verbose': True,
        'cleanup_on_failure': False,
        'save_logs': True
    },
    'ci': {
        'verbose': False,
        'cleanup_on_failure': True,
        'save_logs': True
    },
    'production': {
        'verbose': False,
        'cleanup_on_failure': True,
        'save_logs': False
    }
}

# Get current environment
def get_environment():
    """Get current test environment."""
    env = os.getenv('PILOT_TEST_ENV', 'development')
    return env if env in ENVIRONMENT_CONFIG else 'development'

# Get environment-specific config
def get_env_config():
    """Get environment-specific configuration."""
    env = get_environment()
    return ENVIRONMENT_CONFIG[env]

# Test utilities
def get_executor_config(executor_name):
    """Get configuration for a specific executor."""
    return TEST_CONFIG['executors'].get(executor_name, {})

def get_circuit_config(circuit_name):
    """Get configuration for a specific circuit."""
    return TEST_CONFIG['standard_circuits'].get(circuit_name, {})

def get_performance_thresholds():
    """Get performance thresholds."""
    return TEST_CONFIG['performance_thresholds']

def should_cleanup_on_failure():
    """Check if cleanup should happen on test failure."""
    env_config = get_env_config()
    return env_config.get('cleanup_on_failure', True)

def is_verbose():
    """Check if verbose output is enabled."""
    env_config = get_env_config()
    return env_config.get('verbose', False)

def should_save_logs():
    """Check if logs should be saved."""
    env_config = get_env_config()
    return env_config.get('save_logs', False)
