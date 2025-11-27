#!/usr/bin/env python3
"""
Test to demonstrate executor compatibility and the issue with PennyLane circuits.
"""

import unittest
import sys
import os

# Add the parent directory to the path to import pilot modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pilot.executors.qiskit_executor import QiskitExecutor
from pilot.executors.pennylane_executor import PennylaneExecutor


class TestExecutorCompatibility(unittest.TestCase):
    """Test executor compatibility with different circuit types."""
    
    def test_qiskit_executor_with_pennylane_circuit(self):
        """Test if Qiskit executor can execute a PennyLane circuit."""
        
        # Create a PennyLane circuit function (like in the example)
        def create_pennylane_circuit():
            import pennylane as qml
            
            # Create device (this is the problem!)
            dev = qml.device("default.qubit", wires=2)
            
            @qml.qnode(dev)
            def bell_state():
                qml.Hadamard(wires=0)
                qml.CNOT(wires=[0, 1])
                return qml.probs(wires=[0, 1])
            
            return bell_state
        
        # Create Qiskit executor
        qiskit_executor = QiskitExecutor("qiskit_test", {
            'backend': 'qasm_simulator',
            'shots': 1000
        })
        
        # Try to execute PennyLane circuit with Qiskit executor
        pennylane_circuit = create_pennylane_circuit()
        
        print(f"PennyLane circuit type: {type(pennylane_circuit)}")
        print(f"Is callable: {callable(pennylane_circuit)}")
        
        # This should work because Qiskit executor handles callable circuits
        try:
            result = qiskit_executor.execute_circuit(pennylane_circuit)
            print(f"✅ Qiskit executor successfully executed PennyLane circuit!")
            print(f"Result type: {type(result)}")
            print(f"Result: {result}")
        except Exception as e:
            print(f"❌ Qiskit executor failed to execute PennyLane circuit: {e}")
            # This is expected to fail because the PennyLane circuit is already bound to a device
    
    def test_pennylane_executor_with_pennylane_circuit(self):
        """Test PennyLane executor with PennyLane circuit."""
        
        # Create a PennyLane circuit function
        def create_pennylane_circuit():
            import pennylane as qml
            
            # Create device
            dev = qml.device("default.qubit", wires=2)
            
            @qml.qnode(dev)
            def bell_state():
                qml.Hadamard(wires=0)
                qml.CNOT(wires=[0, 1])
                return qml.probs(wires=[0, 1])
            
            return bell_state
        
        # Create PennyLane executor
        pennylane_executor = PennylaneExecutor("pennylane_test", {
            'device': 'default.qubit',
            'wires': 2,
            'shots': 1000
        })
        
        # Execute PennyLane circuit with PennyLane executor
        pennylane_circuit = create_pennylane_circuit()
        
        try:
            result = pennylane_executor.execute_circuit(pennylane_circuit)
            print(f"✅ PennyLane executor successfully executed PennyLane circuit!")
            print(f"Result type: {type(result)}")
            print(f"Result: {result}")
        except Exception as e:
            print(f"❌ PennyLane executor failed to execute PennyLane circuit: {e}")
    
    def test_circuit_without_device_creation(self):
        """Test what happens when circuit doesn't create its own device."""
        
        # Create a circuit function that doesn't create a device
        def create_circuit_without_device():
            import pennylane as qml
            
            # Just return the circuit definition, not a QNode
            def bell_state():
                qml.Hadamard(wires=0)
                qml.CNOT(wires=[0, 1])
                return qml.probs(wires=[0, 1])
            
            return bell_state  # Raw function, not QNode
        
        # Create Qiskit executor
        qiskit_executor = QiskitExecutor("qiskit_raw_test", {
            'backend': 'qasm_simulator',
            'shots': 1000
        })
        
        # Try to execute raw circuit function
        raw_circuit = create_circuit_without_device()
        
        print(f"Raw circuit type: {type(raw_circuit)}")
        print(f"Is callable: {callable(raw_circuit)}")
        
        try:
            result = qiskit_executor.execute_circuit(raw_circuit)
            print(f"✅ Qiskit executor executed raw circuit function!")
            print(f"Result: {result}")
        except Exception as e:
            print(f"❌ Qiskit executor failed to execute raw circuit: {e}")
            print(f"Error type: {type(e)}")
    
    def test_quantum_execution_service_logic(self):
        """Test how quantum execution service determines executor type."""
        
        from pilot.services.quantum_execution_service import QuantumExecutionService
        
        service = QuantumExecutionService()
        
        # Test resource name parsing
        test_cases = [
            ("qiskit_aer_simulator", "qiskit"),
            ("pennylane_default_device", "pennylane"),
            ("ibmq_fake_backend", "ibmq"),
            ("unknown_backend", "qiskit"),  # Default
        ]
        
        for resource_name, expected_executor in test_cases:
            # Create a mock resource object
            class MockResource:
                def __init__(self, name):
                    self.name = name
            
            resource = MockResource(resource_name)
            
            # Use reflection to test the private method
            executor_type = service._get_executor_type_from_resource(resource)
            
            print(f"Resource: {resource_name} -> Executor: {executor_type} (expected: {expected_executor})")
            self.assertEqual(executor_type, expected_executor)


if __name__ == '__main__':
    unittest.main(verbosity=2)
