#!/usr/bin/env python3
"""
Comprehensive unit tests for all executors in Pilot Quantum framework.

This test suite runs standard workloads against all available executors
and asserts for successful execution.
"""

import unittest
import sys
import os
import time
import tempfile
import shutil

# Add the parent directory to the path to import pilot modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pilot.pilot_compute_service import PilotComputeService, ExecutionEngine
from pilot.dreamer import QuantumTask
from pilot.util.quantum_resource_generator import QuantumResourceGenerator
from tests.test_config import TEST_CONFIG

class TestAllExecutors(unittest.TestCase):
    """Test all executors with standard workloads."""
    
    def setUp(self):
        """Set up test environment."""
        self.working_directory = tempfile.mkdtemp()
        self.pcs = None
        
    def tearDown(self):
        """Clean up test environment."""
        if self.pcs:
            try:
                self.pcs.cancel()
            except:
                pass
        if os.path.exists(self.working_directory):
            shutil.rmtree(self.working_directory)
    
    def create_standard_circuits(self):
        """Create standard test circuits for different executors."""
        circuits = {}
        
        # Qiskit circuits
        def create_qiskit_bell_state():
            from qiskit import QuantumCircuit
            qc = QuantumCircuit(2, 2)
            qc.h(0)
            qc.cx(0, 1)
            return qc
        
        def create_qiskit_ghz_state():
            from qiskit import QuantumCircuit
            qc = QuantumCircuit(3, 3)
            qc.h(0)
            qc.cx(0, 1)
            qc.cx(1, 2)
            return qc
        
        circuits['qiskit'] = {
            'bell_state': create_qiskit_bell_state,
            'ghz_state': create_qiskit_ghz_state
        }
        
        # PennyLane circuits
        def create_pennylane_bell_state():
            import pennylane as qml
            def bell_state():
                qml.Hadamard(wires=0)
                qml.CNOT(wires=[0, 1])
                return qml.probs(wires=[0, 1])
            return bell_state
        
        def create_pennylane_ghz_state():
            import pennylane as qml
            def ghz_state():
                qml.Hadamard(wires=0)
                qml.CNOT(wires=[0, 1])
                qml.CNOT(wires=[1, 2])
                return qml.probs(wires=[0, 1, 2])
            return ghz_state
        
        circuits['pennylane'] = {
            'bell_state': create_pennylane_bell_state,
            'ghz_state': create_pennylane_ghz_state
        }
        
        return circuits
    
    def test_initialization(self):
        """Test PCS initialization."""
        self.pcs = PilotComputeService(
            execution_engine=ExecutionEngine.RAY,
            working_directory=self.working_directory
        )
        self.assertIsNotNone(self.pcs)
    
    def test_qiskit_executor(self):
        """Test Qiskit executor with standard workloads."""
        self.pcs = PilotComputeService(
            execution_engine=ExecutionEngine.RAY,
            working_directory=self.working_directory
        )
        
        # Create pilot with Qiskit executor
        pilot_description = {
            "resource_type": "quantum",
            "quantum": {
                "executor": "qiskit_local"
            },
            "working_directory": self.working_directory,
            "type": "ray",
            "dreamer_enabled": True,
            "resource": "ssh://localhost",
            "cores_per_node": 1
        }
        
        pilot = self.pcs.create_pilot(pilot_description)
        self.assertIsNotNone(pilot)
        
        # Initialize QDREAMER
        qdreamer = self.pcs.initialize_dreamer()
        self.assertIsNotNone(qdreamer)
        
        # Test circuits
        circuits = self.create_standard_circuits()
        qiskit_circuits = circuits['qiskit']
        
        for circuit_name, circuit_func in qiskit_circuits.items():
            with self.subTest(circuit=circuit_name):
                circuit = circuit_func()
                task = QuantumTask(
                    circuit=circuit,
                    num_qubits=2 if 'bell' in circuit_name else 3,
                    gate_set=["h", "cx"],
                    resource_config={}
                )
                
                task_id = self.pcs.submit_quantum_task(task)
                self.assertIsNotNone(task_id)
                
                self.pcs.wait_tasks([task_id])
                result = self.pcs.get_results([task_id])
                self.assertIsNotNone(result)
    
    def test_pennylane_executor(self):
        """Test PennyLane executor with standard workloads."""
        self.pcs = PilotComputeService(
            execution_engine=ExecutionEngine.RAY,
            working_directory=self.working_directory
        )
        
        # Create pilot with PennyLane executor
        pilot_description = {
            "resource_type": "quantum",
            "quantum": {
                "executor": "pennylane",
                "device": "default.qubit"
            },
            "working_directory": self.working_directory,
            "type": "ray",
            "dreamer_enabled": True,
            "resource": "ssh://localhost",
            "cores_per_node": 1
        }
        
        pilot = self.pcs.create_pilot(pilot_description)
        self.assertIsNotNone(pilot)
        
        # Initialize QDREAMER
        qdreamer = self.pcs.initialize_dreamer()
        self.assertIsNotNone(qdreamer)
        
        # Test circuits
        circuits = self.create_standard_circuits()
        pennylane_circuits = circuits['pennylane']
        
        for circuit_name, circuit_func in pennylane_circuits.items():
            with self.subTest(circuit=circuit_name):
                circuit = circuit_func()
                task = QuantumTask(
                    circuit=circuit,
                    num_qubits=2 if 'bell' in circuit_name else 3,
                    gate_set=["h", "cnot"],
                    resource_config={}
                )
                
                task_id = self.pcs.submit_quantum_task(task)
                self.assertIsNotNone(task_id)
                
                self.pcs.wait_tasks([task_id])
                result = self.pcs.get_results([task_id])
                self.assertIsNotNone(result)
    
    def test_multi_executor(self):
        """Test multiple executors in the same pilot."""
        self.pcs = PilotComputeService(
            execution_engine=ExecutionEngine.RAY,
            working_directory=self.working_directory
        )
        
        # Create pilot with multiple executors
        pilot_description = {
            "resource_type": "quantum",
            "quantum": {
                "executor": "qiskit_local"
            },
            "additional_executors": {
                "pennylane": {
                    "device": "default.qubit"
                }
            },
            "working_directory": self.working_directory,
            "type": "ray",
            "dreamer_enabled": True,
            "resource": "ssh://localhost",
            "cores_per_node": 1
        }
        
        pilot = self.pcs.create_pilot(pilot_description)
        self.assertIsNotNone(pilot)
        
        # Initialize QDREAMER
        qdreamer = self.pcs.initialize_dreamer()
        self.assertIsNotNone(qdreamer)
        
        # Verify multiple resources are available
        resources = self.pcs.quantum_resources
        self.assertGreater(len(resources), 1)
    
    def test_resource_selection(self):
        """Test QDREAMER resource selection."""
        self.pcs = PilotComputeService(
            execution_engine=ExecutionEngine.RAY,
            working_directory=self.working_directory
        )
        
        # Create multiple pilots with different executors
        pilot1_description = {
            "resource_type": "quantum",
            "quantum": {
                "executor": "qiskit_local"
            },
            "working_directory": self.working_directory,
            "type": "ray",
            "dreamer_enabled": True,
            "resource": "ssh://localhost",
            "cores_per_node": 1
        }
        
        pilot2_description = {
            "resource_type": "quantum",
            "quantum": {
                "executor": "pennylane",
                "device": "default.qubit"
            },
            "working_directory": self.working_directory,
            "type": "ray",
            "dreamer_enabled": True,
            "resource": "ssh://localhost",
            "cores_per_node": 1
        }
        
        pilot1 = self.pcs.create_pilot(pilot1_description)
        pilot2 = self.pcs.create_pilot(pilot2_description)
        
        # Initialize QDREAMER
        qdreamer = self.pcs.initialize_dreamer()
        self.assertIsNotNone(qdreamer)
        
        # Verify resources from both pilots are available
        resources = self.pcs.quantum_resources
        self.assertGreaterEqual(len(resources), 2)
    
    def test_error_handling(self):
        """Test error handling for invalid configurations."""
        self.pcs = PilotComputeService(
            execution_engine=ExecutionEngine.RAY,
            working_directory=self.working_directory
        )
        
        # Test invalid executor
        with self.assertRaises(Exception):
            pilot_description = {
                "resource_type": "quantum",
                "quantum": {
                    "executor": "invalid_executor"
                },
                "working_directory": self.working_directory,
                "type": "ray",
                "dreamer_enabled": True,
                "resource": "ssh://localhost",
                "cores_per_node": 1
            }
            pilot = self.pcs.create_pilot(pilot_description)
    
    def test_circuit_validation(self):
        """Test circuit validation."""
        self.pcs = PilotComputeService(
            execution_engine=ExecutionEngine.RAY,
            working_directory=self.working_directory
        )
        
        # Create pilot
        pilot_description = {
            "resource_type": "quantum",
            "quantum": {
                "executor": "qiskit_local"
            },
            "working_directory": self.working_directory,
            "type": "ray",
            "dreamer_enabled": True,
            "resource": "ssh://localhost",
            "cores_per_node": 1
        }
        
        pilot = self.pcs.create_pilot(pilot_description)
        qdreamer = self.pcs.initialize_dreamer()
        
        # Test invalid circuit
        with self.assertRaises(Exception):
            task = QuantumTask(
                circuit="invalid_circuit",
                num_qubits=2,
                gate_set=["h", "cx"],
                resource_config={}
            )
            task_id = self.pcs.submit_quantum_task(task)


class TestExecutorPerformance(unittest.TestCase):
    """Test executor performance."""
    
    def setUp(self):
        """Set up test environment."""
        self.working_directory = tempfile.mkdtemp()
        self.pcs = None
        
    def tearDown(self):
        """Clean up test environment."""
        if self.pcs:
            try:
                self.pcs.cancel()
            except:
                pass
        if os.path.exists(self.working_directory):
            shutil.rmtree(self.working_directory)
    
    def test_execution_time(self):
        """Test execution time for different executors."""
        executors = ['qiskit_local', 'pennylane']
        execution_times = {}
        
        for executor in executors:
            self.pcs = PilotComputeService(
                execution_engine=ExecutionEngine.RAY,
                working_directory=self.working_directory
            )
            
            # Create pilot
            pilot_description = {
                "resource_type": "quantum",
                "quantum": {
                    "executor": executor
                },
                "working_directory": self.working_directory,
                "type": "ray",
                "dreamer_enabled": True,
                "resource": "ssh://localhost",
                "cores_per_node": 1
            }
            
            if executor == 'pennylane':
                pilot_description["quantum"]["device"] = "default.qubit"
            
            pilot = self.pcs.create_pilot(pilot_description)
            qdreamer = self.pcs.initialize_dreamer()
            
            # Create test circuit
            if executor == 'qiskit_local':
                from qiskit import QuantumCircuit
                circuit = QuantumCircuit(2, 2)
                circuit.h(0)
                circuit.cx(0, 1)
            else:  # pennylane
                import pennylane as qml
                def bell_state():
                    qml.Hadamard(wires=0)
                    qml.CNOT(wires=[0, 1])
                    return qml.probs(wires=[0, 1])
                circuit = bell_state
            
            # Measure execution time
            start_time = time.time()
            task = QuantumTask(
                circuit=circuit,
                num_qubits=2,
                gate_set=["h", "cx"] if executor == 'qiskit_local' else ["h", "cnot"],
                resource_config={}
            )
            
            task_id = self.pcs.submit_quantum_task(task)
            self.pcs.wait_tasks([task_id])
            result = self.pcs.get_results([task_id])
            
            end_time = time.time()
            execution_times[executor] = end_time - start_time
            
            # Cleanup
            self.pcs.cancel()
        
        # Verify execution times are reasonable
        for executor, exec_time in execution_times.items():
            self.assertLess(exec_time, 60.0, f"{executor} took too long: {exec_time:.2f}s")
            self.assertGreater(exec_time, 0.1, f"{executor} was too fast: {exec_time:.2f}s")


if __name__ == '__main__':
    unittest.main(verbosity=2)
