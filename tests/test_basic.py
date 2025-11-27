#!/usr/bin/env python3
"""
Basic tests for Pilot Quantum framework.

This module contains basic tests to verify that the framework components
can be imported and initialized correctly.
"""

import unittest
import sys
import os

# Add the parent directory to the path to import pilot modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestBasicImports(unittest.TestCase):
    """Test basic imports and initialization."""
    
    def test_pilot_imports(self):
        """Test that pilot modules can be imported."""
        try:
            from pilot.pilot_compute_service import PilotComputeService, ExecutionEngine
            from pilot.dreamer import Q_DREAMER, QuantumTask
            from pilot.util.quantum_resource_generator import QuantumResourceGenerator
            from pilot.executors.qiskit_executor import QiskitExecutor
            from pilot.executors.pennylane_executor import PennylaneExecutor
            from pilot.executors.executor_factory import QuantumExecutorFactory
            self.assertTrue(True, "All pilot modules imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import pilot modules: {e}")
    
    def test_test_config_import(self):
        """Test that test configuration can be imported."""
        try:
            from tests.test_config import TEST_CONFIG
            self.assertIsNotNone(TEST_CONFIG)
            self.assertIn('executors', TEST_CONFIG)
            self.assertIn('circuits', TEST_CONFIG)
        except ImportError as e:
            self.fail(f"Failed to import test configuration: {e}")
    
    def test_quantum_resource_generator_initialization(self):
        """Test QuantumResourceGenerator initialization."""
        try:
            from pilot.util.quantum_resource_generator import QuantumResourceGenerator
            generator = QuantumResourceGenerator()
            self.assertIsNotNone(generator)
        except Exception as e:
            self.fail(f"Failed to initialize QuantumResourceGenerator: {e}")
    
    def test_executor_list(self):
        """Test that we can get a list of available executors."""
        try:
            from pilot.util.quantum_resource_generator import QuantumResourceGenerator
            generator = QuantumResourceGenerator()
            
            # Test getting resources for different executors
            executors = ['qiskit_local', 'pennylane']
            
            for executor in executors:
                with self.subTest(executor=executor):
                    config = None
                    if executor == 'pennylane':
                        config = {'device': 'default.qubit'}
                    
                    resources = generator.get_quantum_resources(executor, config)
                    self.assertIsNotNone(resources)
                    self.assertIsInstance(resources, dict)
        except Exception as e:
            self.fail(f"Failed to get executor list: {e}")
    
    def test_executor_factory(self):
        """Test QuantumExecutorFactory initialization."""
        try:
            from pilot.executors.executor_factory import QuantumExecutorFactory
            factory = QuantumExecutorFactory()
            self.assertIsNotNone(factory)
        except Exception as e:
            self.fail(f"Failed to initialize QuantumExecutorFactory: {e}")
    
    def test_pilot_compute_service_initialization(self):
        """Test PilotComputeService initialization."""
        try:
            from pilot.pilot_compute_service import PilotComputeService, ExecutionEngine
            import tempfile
            
            with tempfile.TemporaryDirectory() as temp_dir:
                pcs = PilotComputeService(
                    execution_engine=ExecutionEngine.RAY,
                    working_directory=temp_dir
                )
                self.assertIsNotNone(pcs)
        except Exception as e:
            self.fail(f"Failed to initialize PilotComputeService: {e}")
    
    def test_quantum_task_creation(self):
        """Test QuantumTask creation."""
        try:
            from pilot.dreamer import QuantumTask
            
            # Test with a simple circuit
            def simple_circuit():
                return "test_circuit"
            
            task = QuantumTask(
                circuit=simple_circuit,
                num_qubits=2,
                gate_set=["h", "cx"],
                resource_config={}
            )
            self.assertIsNotNone(task)
            self.assertEqual(task.num_qubits, 2)
            self.assertEqual(task.gate_set, ["h", "cx"])
        except Exception as e:
            self.fail(f"Failed to create QuantumTask: {e}")
    
    def test_qdreamer_initialization(self):
        """Test Q_DREAMER initialization."""
        try:
            from pilot.dreamer import Q_DREAMER
            from pilot.util.quantum_resource_generator import QuantumResourceGenerator
            
            # Create some test resources
            generator = QuantumResourceGenerator()
            resources = generator.get_quantum_resources('qiskit_local')
            
            # Test with minimal configuration
            config = {
                'load_balancing': {
                    'fidelity_weight': 0.5,
                    'queue_weight': 0.5
                }
            }
            
            qdreamer = Q_DREAMER(config, resources)
            self.assertIsNotNone(qdreamer)
        except Exception as e:
            self.fail(f"Failed to initialize Q_DREAMER: {e}")


class TestConfigurationValidation(unittest.TestCase):
    """Test configuration validation."""
    
    def test_executor_configs(self):
        """Test that executor configurations are valid."""
        from tests.test_config import TEST_CONFIG
        
        executors = TEST_CONFIG['executors']
        self.assertIn('qiskit_local', executors)
        self.assertIn('pennylane', executors)
        
        # Test Qiskit config
        qiskit_config = executors['qiskit_local']
        self.assertIn('backend', qiskit_config)
        self.assertIn('shots', qiskit_config)
        
        # Test PennyLane config
        pennylane_config = executors['pennylane']
        self.assertIn('device', pennylane_config)
        self.assertIn('wires', pennylane_config)
        self.assertIn('shots', pennylane_config)
    
    def test_circuit_configs(self):
        """Test that circuit configurations are valid."""
        from tests.test_config import TEST_CONFIG
        
        circuits = TEST_CONFIG['circuits']
        self.assertIn('bell_state', circuits)
        self.assertIn('ghz_state', circuits)
        
        # Test Bell state config
        bell_config = circuits['bell_state']
        self.assertEqual(bell_config['qubits'], 2)
        self.assertIn('h', bell_config['gates'])
        self.assertIn('cx', bell_config['gates'])
    
    def test_performance_thresholds(self):
        """Test that performance thresholds are reasonable."""
        from tests.test_config import TEST_CONFIG
        
        performance = TEST_CONFIG['performance']
        self.assertGreater(performance['max_execution_time'], 0)
        self.assertGreater(performance['min_execution_time'], 0)
        self.assertLess(performance['min_execution_time'], performance['max_execution_time'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
