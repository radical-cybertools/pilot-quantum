#!/usr/bin/env python3
"""
Simple tests for dreamer.py functionality.
"""

import unittest
from unittest.mock import Mock, patch
from pilot.dreamer import (
    TaskType, DreamerStrategyType, QuantumTask, QuantumResource,
    StrategySelector, RoundRobinStrategy, LeastErrorRateStrategy, LeastBusyStrategy,
    Q_DREAMER
)
from qiskit import QuantumCircuit


class TestDreamer(unittest.TestCase):
    """Test cases for dreamer.py functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test quantum resources
        self.resources = {
            'resource1': QuantumResource(
                name='resource1',
                qubit_count=5,
                gateset=['h', 'cx', 'x', 'z'],
                error_rate=0.01,
                noise_level=0.1
            ),
            'resource2': QuantumResource(
                name='resource2',
                qubit_count=3,
                gateset=['h', 'cx', 'x', 'z'],
                error_rate=0.05,
                noise_level=0.2
            ),
            'resource3': QuantumResource(
                name='resource3',
                qubit_count=4,
                gateset=['h', 'cx', 'x', 'z'],
                error_rate=0.02,
                noise_level=0.15
            )
        }
        
        # Create test quantum circuit
        self.test_circuit = QuantumCircuit(2, 2)
        self.test_circuit.h(0)
        self.test_circuit.cx(0, 1)
        self.test_circuit.measure([0, 1], [0, 1])
        
        # Create test quantum task
        self.quantum_task = QuantumTask(
            circuits=[self.test_circuit],
            resource_config={},
            kwargs={}
        )
    
    def test_task_type_enum(self):
        """Test TaskType enum values."""
        self.assertEqual(TaskType.CLASSICAL.value, "classical")
        self.assertEqual(TaskType.QUANTUM.value, "quantum")
        self.assertEqual(TaskType.HYBRID.value, "hybrid")
    
    def test_dreamer_strategy_type_enum(self):
        """Test DreamerStrategyType enum values."""
        self.assertEqual(DreamerStrategyType.LEAST_ERROR_RATE.value, "least_error_rate")
        self.assertEqual(DreamerStrategyType.ROUND_ROBIN.value, "round_robin")
        self.assertEqual(DreamerStrategyType.LEAST_BUSY.value, "least_busy")
    
    def test_quantum_task_creation(self):
        """Test QuantumTask creation."""
        self.assertEqual(self.quantum_task.type, TaskType.QUANTUM)
        self.assertEqual(len(self.quantum_task.circuits), 1)
        self.assertIsInstance(self.quantum_task.circuits[0], QuantumCircuit)
        self.assertIsNotNone(self.quantum_task.task_id)
    
    def test_quantum_resource_creation(self):
        """Test QuantumResource creation."""
        resource = self.resources['resource1']
        self.assertEqual(resource.name, 'resource1')
        self.assertEqual(resource.qubit_count, 5)
        self.assertEqual(resource.error_rate, 0.01)
        self.assertEqual(resource.noise_level, 0.1)
        self.assertIn('h', resource.gateset)
        self.assertIn('cx', resource.gateset)
    
    def test_least_error_rate_strategy(self):
        """Test LeastErrorRateStrategy selection."""
        strategy = LeastErrorRateStrategy()
        selected_resource = strategy.select_resource(self.quantum_task, self.resources)
        
        # Should select resource with lowest error rate (resource1 with 0.01)
        self.assertEqual(selected_resource.name, 'resource1')
        self.assertEqual(selected_resource.error_rate, 0.01)
    
    def test_round_robin_strategy(self):
        """Test RoundRobinStrategy selection."""
        strategy = RoundRobinStrategy()
        
        # First selection
        resource1 = strategy.select_resource(self.quantum_task, self.resources)
        self.assertIsNotNone(resource1)
        
        # Second selection (should be different)
        resource2 = strategy.select_resource(self.quantum_task, self.resources)
        self.assertIsNotNone(resource2)
        
        # Third selection (should cycle back)
        resource3 = strategy.select_resource(self.quantum_task, self.resources)
        self.assertIsNotNone(resource3)
    
    def test_least_busy_strategy(self):
        """Test LeastBusyStrategy selection."""
        strategy = LeastBusyStrategy()
        selected_resource = strategy.select_resource(self.quantum_task, self.resources)
        
        # Should return a resource (currently returns first available)
        self.assertIsNotNone(selected_resource)
        self.assertIn(selected_resource.name, self.resources.keys())
    
    def test_strategy_with_empty_resources(self):
        """Test strategies with empty resources."""
        empty_resources = {}
        
        least_error_strategy = LeastErrorRateStrategy()
        result = least_error_strategy.select_resource(self.quantum_task, empty_resources)
        self.assertIsNone(result)
        
        round_robin_strategy = RoundRobinStrategy()
        result = round_robin_strategy.select_resource(self.quantum_task, empty_resources)
        self.assertIsNone(result)
        
        least_busy_strategy = LeastBusyStrategy()
        result = least_busy_strategy.select_resource(self.quantum_task, empty_resources)
        self.assertIsNone(result)
    
    def test_q_dreamer_initialization(self):
        """Test Q_DREAMER initialization."""
        dreamer = Q_DREAMER(self.resources, DreamerStrategyType.LEAST_ERROR_RATE)
        
        self.assertEqual(dreamer.quantum_resources, self.resources)
        self.assertIsInstance(dreamer.strategy_selector, LeastErrorRateStrategy)
    
    def test_q_dreamer_get_best_resource(self):
        """Test Q_DREAMER resource selection."""
        dreamer = Q_DREAMER(self.resources, DreamerStrategyType.LEAST_ERROR_RATE)
        
        with patch('builtins.print') as mock_print:
            best_resource = dreamer.get_best_resource(self.quantum_task)
            
            self.assertIsNotNone(best_resource)
            self.assertEqual(best_resource.name, 'resource1')  # Lowest error rate
            mock_print.assert_called_once()
    
    def test_q_dreamer_strategy_factory(self):
        """Test Q_DREAMER strategy factory method."""
        dreamer = Q_DREAMER(self.resources, DreamerStrategyType.ROUND_ROBIN)
        
        strategy = dreamer.get_strategy()
        self.assertIsInstance(strategy, RoundRobinStrategy)
        
        dreamer = Q_DREAMER(self.resources, DreamerStrategyType.LEAST_BUSY)
        strategy = dreamer.get_strategy()
        self.assertIsInstance(strategy, LeastBusyStrategy)
    
    def test_invalid_strategy_type(self):
        """Test invalid strategy type handling."""
        # Create a dreamer with a valid strategy type first
        dreamer = Q_DREAMER(self.resources, DreamerStrategyType.LEAST_ERROR_RATE)
        
        # Now manually set an invalid strategy type to test error handling
        dreamer.strategy_type = "invalid_strategy"
        
        with self.assertRaises(ValueError):
            dreamer.get_strategy()


class TestQuantumCircuitIntegration(unittest.TestCase):
    """Test quantum circuit integration."""
    
    def test_quantum_circuit_creation(self):
        """Test quantum circuit creation and properties."""
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])
        
        self.assertEqual(qc.num_qubits, 2)
        self.assertEqual(qc.num_clbits, 2)
        self.assertEqual(qc.depth(), 3)  # h + cx + measure = 3 layers
    
    def test_quantum_task_with_multiple_circuits(self):
        """Test QuantumTask with multiple circuits."""
        circuits = []
        for i in range(3):
            qc = QuantumCircuit(2, 2)
            qc.h(0)
            qc.cx(0, 1)
            qc.measure([0, 1], [0, 1])
            circuits.append(qc)
        
        task = QuantumTask(circuits=circuits, resource_config={}, kwargs={})
        
        self.assertEqual(len(task.circuits), 3)
        self.assertEqual(task.type, TaskType.QUANTUM)


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)
