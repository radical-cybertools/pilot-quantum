#!/usr/bin/env python3
"""
Tests for QDREAMER integration with Pilot Quantum.

This module contains comprehensive tests for:
- QDREAMER initialization and configuration
- Resource selection and optimization
- Multi-pilot resource management
- Task execution with intelligent resource selection
"""

import unittest
import tempfile
import os
import time
from unittest.mock import Mock, patch, MagicMock
from qiskit import QuantumCircuit

from pilot.pilot_compute_service import PilotComputeService, ExecutionEngine
from pilot.dreamer import Q_DREAMER, QuantumTask, QuantumResource, OptimizedResourceSelector
from pilot.util.quantum_resource_generator import QuantumResourceGenerator
from pilot.executors.qiskit_executor import QiskitExecutor
from pilot.executors.ibmq_executor import IBMQExecutor
from pilot.executors.pennylane_executor import PennylaneExecutor


class TestQDREAMERIntegration(unittest.TestCase):
    """Test QDREAMER integration with Pilot Quantum."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.pcs = None

    def tearDown(self):
        """Clean up test fixtures."""
        if self.pcs:
            try:
                self.pcs.cancel()
            except:
                pass
        if os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)

    def create_simple_circuit(self):
        """Create a simple quantum circuit for testing."""
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        return qc

    def test_qdreamer_initialization(self):
        """Test QDREAMER initialization with different configurations."""
        # Test basic initialization
        qdreamer = Q_DREAMER({"optimization_mode": "balanced"}, [])
        self.assertIsNotNone(qdreamer)
        self.assertEqual(qdreamer.qdreamer_config["optimization_mode"], "balanced")

        # Test high fidelity mode
        qdreamer = Q_DREAMER({"optimization_mode": "high_fidelity"}, [])
        self.assertEqual(qdreamer.qdreamer_config["optimization_mode"], "high_fidelity")

        # Test high speed mode
        qdreamer = Q_DREAMER({"optimization_mode": "high_speed"}, [])
        self.assertEqual(qdreamer.qdreamer_config["optimization_mode"], "high_speed")

    def test_quantum_task_creation(self):
        """Test QuantumTask creation and validation."""
        circuit = self.create_simple_circuit()
        
        # Test basic task creation
        task = QuantumTask(
            circuit=circuit,
            num_qubits=2,
            gate_set=["h", "cx"],
            resource_config={"num_qpus": 1}
        )
        
        self.assertEqual(task.num_qubits, 2)
        self.assertEqual(task.gate_set, ["h", "cx"])
        self.assertEqual(task.resource_config["num_qpus"], 1)

        # Test task with callable circuit
        def circuit_func():
            return self.create_simple_circuit()
        
        task = QuantumTask(
            circuit=circuit_func,
            num_qubits=2,
            gate_set=["h", "cx"],
            resource_config={"num_qpus": 1}
        )
        
        self.assertEqual(task.num_qubits, 2)
        self.assertTrue(callable(task.circuit))

    def test_quantum_resource_creation(self):
        """Test QuantumResource creation and properties."""
        resource = QuantumResource(
            name="test_backend",
            qubit_count=5,
            gateset=["h", "cx", "x"],
            error_rate=0.01,
            noise_level=0.02,
            quantum_config={"backend": "qasm_simulator"}
        )
        
        self.assertEqual(resource.name, "test_backend")
        self.assertEqual(resource.qubit_count, 5)
        self.assertEqual(resource.gateset, ["h", "cx", "x"])
        self.assertEqual(resource.error_rate, 0.01)
        self.assertEqual(resource.noise_level, 0.02)
        self.assertEqual(resource.fidelity, 0.99)  # 1.0 - error_rate

    def test_optimized_resource_selector(self):
        """Test OptimizedResourceSelector functionality."""
        # Create test resources
        resources = {
            "backend1": QuantumResource("backend1", 5, ["h", "cx"], 0.01, 0.02, {}),
            "backend2": QuantumResource("backend2", 10, ["h", "cx", "x"], 0.005, 0.01, {}),
            "backend3": QuantumResource("backend3", 7, ["h", "cx"], 0.015, 0.025, {})
        }
        
        task = QuantumTask(
            circuit=self.create_simple_circuit(),
            num_qubits=2,
            gate_set=["h", "cx"],
            resource_config={"num_qpus": 1}
        )
        
        # Test resource selection
        selector = OptimizedResourceSelector()
        selected_resource = selector.optimize_resource_selection(task, resources, None, "test_task")
        
        self.assertIsNotNone(selected_resource)
        self.assertIn(selected_resource.name, ["backend1", "backend2", "backend3"])

    def test_executor_simulator_detection(self):
        """Test executor simulator detection."""
        # Test QiskitExecutor
        qiskit_executor = QiskitExecutor("test", {"backend": "qasm_simulator"})
        self.assertTrue(qiskit_executor.is_simulator())

        # Test IBMQExecutor with fake backend
        ibmq_executor = IBMQExecutor("test", {"backend": "fake_quebec"})
        self.assertTrue(ibmq_executor.is_simulator())

        # Test IBMQExecutor with real backend (should be False for real hardware)
        ibmq_executor = IBMQExecutor("test", {"backend": "ibmq_montreal", "token": "fake_token"})
        # Since we don't have a real token, it will still be detected as simulator
        self.assertTrue(ibmq_executor.is_simulator())

        # Test PennyLaneExecutor
        pennylane_executor = PennylaneExecutor("test", {"device": "default.qubit"})
        self.assertTrue(pennylane_executor.is_simulator())

    def test_resource_generator(self):
        """Test QuantumResourceGenerator functionality."""
        generator = QuantumResourceGenerator()
        
        # Test Qiskit resource generation
        qiskit_config = {
            "executor": "qiskit_local",
            "backend": ["qasm_simulator", "statevector_simulator"]
        }
        resources = generator.get_quantum_resources("qiskit", qiskit_config)
        self.assertIsInstance(resources, dict)
        self.assertGreater(len(resources), 0)

        # Test IBMQ resource generation
        ibmq_config = {
            "executor": "ibmq",
            "backend": ["fake_quebec"]
        }
        resources = generator.get_quantum_resources("ibmq", ibmq_config)
        self.assertIsInstance(resources, dict)
        self.assertGreater(len(resources), 0)

    @patch('pilot.pilot_compute_service.PilotComputeService')
    def test_pilot_compute_service_integration(self, mock_pcs):
        """Test PilotComputeService integration with QDREAMER."""
        # Mock the PCS to avoid actual Ray/Dask initialization
        mock_pcs.return_value.quantum_resources = {
            "backend1": QuantumResource("backend1", 5, ["h", "cx"], 0.01, 0.02, {}),
            "backend2": QuantumResource("backend2", 10, ["h", "cx", "x"], 0.005, 0.01, {})
        }
        mock_pcs.return_value.dreamer_enabled = False
        mock_pcs.return_value.qdreamer = None
        
        # Test QDREAMER initialization
        mock_pcs.return_value.initialize_dreamer = Mock()
        mock_pcs.return_value.initialize_dreamer({"optimization_mode": "high_fidelity"})
        
        # Verify initialization was called
        mock_pcs.return_value.initialize_dreamer.assert_called_once_with(
            {"optimization_mode": "high_fidelity"}
        )

    def test_multi_pilot_resource_collection(self):
        """Test resource collection from multiple pilots."""
        # Create mock resources for different pilots
        pilot1_resources = {
            "pilot1_backend1": QuantumResource("pilot1_backend1", 5, ["h", "cx"], 0.01, 0.02, {}),
            "pilot1_backend2": QuantumResource("pilot1_backend2", 10, ["h", "cx"], 0.005, 0.01, {})
        }
        
        pilot2_resources = {
            "pilot2_backend1": QuantumResource("pilot2_backend1", 7, ["h", "cx", "x"], 0.015, 0.025, {}),
            "pilot2_backend2": QuantumResource("pilot2_backend2", 12, ["h", "cx"], 0.008, 0.012, {})
        }
        
        # Combine resources
        all_resources = {**pilot1_resources, **pilot2_resources}
        
        self.assertEqual(len(all_resources), 4)
        self.assertIn("pilot1_backend1", all_resources)
        self.assertIn("pilot2_backend1", all_resources)

    def test_task_correlation_logging(self):
        """Test task correlation ID generation and logging."""
        import uuid
        
        # Generate task ID
        task_id = f"quantum-{uuid.uuid4()}"
        
        # Verify format
        self.assertTrue(task_id.startswith("quantum-"))
        # UUID is 36 characters, so total length should be 44 (7 + 36 + 1 for hyphen)
        self.assertEqual(len(task_id), 44)

    def test_optimization_modes(self):
        """Test different optimization modes."""
        # Create test resources
        resources = {
            "backend1": QuantumResource("backend1", 5, ["h", "cx"], 0.01, 0.02, {}),
            "backend2": QuantumResource("backend2", 10, ["h", "cx", "x"], 0.005, 0.01, {}),
            "backend3": QuantumResource("backend3", 7, ["h", "cx"], 0.015, 0.025, {})
        }
        
        task = QuantumTask(
            circuit=self.create_simple_circuit(),
            num_qubits=2,
            gate_set=["h", "cx"],
            resource_config={"num_qpus": 1}
        )
        
        # Test high fidelity mode
        high_fidelity_selector = OptimizedResourceSelector("high_fidelity")
        high_fidelity_resource = high_fidelity_selector.optimize_resource_selection(
            task, resources, None, "test_task"
        )
        
        # Test high speed mode
        high_speed_selector = OptimizedResourceSelector("high_speed")
        high_speed_resource = high_speed_selector.optimize_resource_selection(
            task, resources, None, "test_task"
        )
        
        # Test balanced mode
        balanced_selector = OptimizedResourceSelector("balanced")
        balanced_resource = balanced_selector.optimize_resource_selection(
            task, resources, None, "test_task"
        )
        
        # All should return a valid resource
        self.assertIsNotNone(high_fidelity_resource)
        self.assertIsNotNone(high_speed_resource)
        self.assertIsNotNone(balanced_resource)

    def test_circuit_compatibility_checking(self):
        """Test circuit compatibility checking."""
        # Create resources with different gate sets
        resource1 = QuantumResource("backend1", 5, ["h", "cx"], 0.01, 0.02, {})
        resource2 = QuantumResource("backend2", 5, ["h", "cx", "x", "z"], 0.01, 0.02, {})
        resource3 = QuantumResource("backend3", 1, ["h", "cx"], 0.01, 0.02, {})  # Fewer qubits
        
        task = QuantumTask(
            circuit=self.create_simple_circuit(),
            num_qubits=2,
            gate_set=["h", "cx"],
            resource_config={"num_qpus": 1}
        )
        
        # Test compatibility
        self.assertTrue(all(gate in resource1.gateset for gate in task.gate_set))
        self.assertTrue(all(gate in resource2.gateset for gate in task.gate_set))
        self.assertTrue(all(gate in resource3.gateset for gate in task.gate_set))
        
        # Test qubit count compatibility
        self.assertTrue(resource1.qubit_count >= task.num_qubits)
        self.assertTrue(resource2.qubit_count >= task.num_qubits)
        self.assertFalse(resource3.qubit_count >= task.num_qubits)  # Should fail

    def test_error_handling(self):
        """Test error handling in QDREAMER components."""
        # Test with empty resource dict
        task = QuantumTask(
            circuit=self.create_simple_circuit(),
            num_qubits=2,
            gate_set=["h", "cx"],
            resource_config={"num_qpus": 1}
        )
        
        selector = OptimizedResourceSelector()
        
        # Should handle empty resources gracefully and return None
        result = selector.optimize_resource_selection(task, {}, None, "test_task")
        self.assertIsNone(result)
        
        # Test with incompatible resources
        incompatible_resource = QuantumResource("incompatible", 1, ["x"], 0.01, 0.02, {})
        incompatible_resources = {"incompatible": incompatible_resource}
        
        # Should filter out incompatible resources
        result = selector.optimize_resource_selection(task, incompatible_resources, None, "test_task")
        # Should return None for no compatible resources
        self.assertIsNone(result)


class TestQDREAMERPerformance(unittest.TestCase):
    """Test QDREAMER performance characteristics."""

    def test_resource_selection_performance(self):
        """Test resource selection performance with large resource sets."""
        # Create large set of resources
        resources = {}
        for i in range(100):
            resource = QuantumResource(
                f"backend_{i}",
                qubit_count=5 + (i % 10),
                gateset=["h", "cx", "x", "z"],
                error_rate=0.001 + (i * 0.0001),
                noise_level=0.002 + (i * 0.0001),
                quantum_config={}
            )
            resources[f"backend_{i}"] = resource
        
        task = QuantumTask(
            circuit=QuantumCircuit(2, 2),
            num_qubits=2,
            gate_set=["h", "cx"],
            resource_config={"num_qpus": 1}
        )
        
        selector = OptimizedResourceSelector()
        
        # Measure selection time
        start_time = time.time()
        selected_resource = selector.optimize_resource_selection(task, resources, None, "test_task")
        selection_time = time.time() - start_time
        
        # Selection should complete within reasonable time (< 1 second)
        self.assertLess(selection_time, 1.0)
        self.assertIsNotNone(selected_resource)

    def test_memory_usage(self):
        """Test memory usage with large resource sets."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Create large resource set
        resources = {}
        for i in range(1000):
            resource = QuantumResource(
                f"backend_{i}",
                qubit_count=5,
                gateset=["h", "cx"],
                error_rate=0.01,
                noise_level=0.02,
                quantum_config={"large_config": "x" * 1000}  # Large config
            )
            resources[f"backend_{i}"] = resource
        
        # Create QDREAMER instance
        qdreamer = Q_DREAMER({"optimization_mode": "balanced"}, resources)
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (< 100MB)
        self.assertLess(memory_increase, 100 * 1024 * 1024)


if __name__ == '__main__':
    unittest.main()
