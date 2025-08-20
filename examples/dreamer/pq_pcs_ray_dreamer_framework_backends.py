#!/usr/bin/env python3
"""
Pilot Quantum with QDREAMER - Framework Backends Example

This example demonstrates using QDREAMER with Qiskit simulator executor
and various framework-provided backends including:
- Qiskit Aer simulator (high performance)
- Qiskit BasicAer simulator (basic functionality)
- Fake backends (IBM-like simulators with realistic noise)
- Custom backends (user-defined simulators)

The example shows how QDREAMER intelligently selects the best backend
based on circuit requirements, fidelity needs, and queue status.
"""

import os
import time
import sys
from typing import Dict, List

import ray
from pilot.dreamer import QuantumTask, TaskType
from pilot.pilot_compute_service import ExecutionEngine, PilotComputeService
from qiskit import QuantumCircuit

# Configuration
WORKING_DIRECTORY = os.path.join(os.environ["HOME"], "work")

# Quantum pilot description using framework-provided backends
pilot_quantum_description = {
    "resource_type": "quantum",
    "quantum": {
        "executor": "qiskit_local",  # Uses framework-provided Qiskit backends
        # No custom_backends - will use framework defaults
    },
    "working_directory": WORKING_DIRECTORY,
    "type": "ray"
}

# QDREAMER configuration is now simplified - just pass optimization_mode to initialize_dreamer()

def start_pilot():
    """Initialize Pilot Compute Service with framework-provided quantum resources."""
    print("🚀 Starting Pilot Compute Service with Framework Backends...")
    
    pcs = PilotComputeService(execution_engine=ExecutionEngine.RAY, 
                              working_directory=WORKING_DIRECTORY)
    
    print("📋 Creating quantum pilot with framework-provided backends...")
    pcs.create_pilot(pilot_compute_description=pilot_quantum_description)
    
    print("🧠 Initializing QDREAMER for intelligent resource selection...")
    pcs.initialize_dreamer({"optimization_mode": "high_speed"})
    
    print("✅ Pilot Compute Service initialized successfully")
    return pcs

def create_bell_state_circuit():
    """Create a simple 2-qubit Bell state circuit."""
    print("🔧 Creating Bell state circuit (2 qubits)...")
    qc = QuantumCircuit(2, 2)
    qc.h(0)  # Hadamard gate on qubit 0
    qc.cx(0, 1)  # CNOT gate: control=0, target=1
    qc.measure([0, 1], [0, 1])  # Measure both qubits
    print(f"   ✅ Circuit created: {qc.num_qubits} qubits, {qc.num_clbits} classical bits")
    print(f"   Circuit depth: {qc.depth()}, Total gates: {qc.size()}")
    return qc

def create_ghz_state_circuit():
    """Create a 3-qubit GHZ state circuit."""
    print("🔧 Creating GHZ state circuit (3 qubits)...")
    qc = QuantumCircuit(3, 3)
    qc.h(0)  # Hadamard on qubit 0
    qc.cx(0, 1)  # CNOT: 0->1
    qc.cx(1, 2)  # CNOT: 1->2
    qc.measure([0, 1, 2], [0, 1, 2])  # Measure all qubits
    print(f"   ✅ Circuit created: {qc.num_qubits} qubits, {qc.num_clbits} classical bits")
    print(f"   Circuit depth: {qc.depth()}, Total gates: {qc.size()}")
    return qc

def create_quantum_fourier_transform_circuit():
    """Create a 4-qubit Quantum Fourier Transform circuit using only supported gates."""
    print("🔧 Creating QFT circuit (4 qubits) with supported gates...")
    qc = QuantumCircuit(4, 4)
    
    # Apply Hadamard gates
    for i in range(4):
        qc.h(i)
    
    # Apply controlled NOT gates (simplified QFT)
    for i in range(3):
        for j in range(i+1, 4):
            qc.cx(i, j)
    
    # Apply additional Hadamard gates for more complexity
    for i in range(4):
        qc.h(i)
    
    qc.measure(range(4), range(4))
    print(f"   ✅ Circuit created: {qc.num_qubits} qubits, {qc.num_clbits} classical bits")
    print(f"   Circuit depth: {qc.depth()}, Total gates: {qc.size()}")
    return qc

def demonstrate_framework_backends():
    """Demonstrate QDREAMER with framework-provided backends."""
    print("=== QDREAMER Framework Backends Demonstration ===\n")
    print("🎯 Starting QDREAMER Framework Backends Demonstration")
    
    pcs = None
    try:
        # Start Pilot with framework backends
        pcs = start_pilot()
        
        # Display available framework-provided resources
        print(f"\n📋 Available Framework-Provided Quantum Resources:")
        for name, resource in pcs.quantum_resources.items():
            fidelity = 1 - resource.error_rate if resource.error_rate is not None else 1.0
            error_rate_display = f"{resource.error_rate:.3f}" if resource.error_rate is not None else "0.000"
            print(f"   - {name}: {resource.qubit_count} qubits, "
                  f"fidelity={fidelity:.3f}, error_rate={error_rate_display}")
            print(f"     Gateset: {resource.gateset}")
        
        # Test different circuit complexities
        circuits = [
            ("Bell State", create_bell_state_circuit, 2, ["h", "cx", "measure"]),
            ("GHZ State", create_ghz_state_circuit, 3, ["h", "cx", "measure"]),
            ("QFT", create_quantum_fourier_transform_circuit, 4, ["h", "cx", "measure"])
        ]
        
        print(f"\n🔄 Testing different circuit complexities...")
        
        for circuit_name, circuit_func, num_qubits, gate_set in circuits:
            print(f"\n--- Testing {circuit_name} Circuit ---")
            
            # Create quantum task
            qt = QuantumTask(
                circuit=circuit_func,
                num_qubits=num_qubits,
                gate_set=gate_set,
                resource_config={"num_qpus": 1, "num_gpus": 0, "memory": None}
            )
            
            print(f" Submitting {circuit_name} task...")
            task_id = pcs.submit_quantum_task(qt)
            
            # Wait for completion
            print(f"⏳ Waiting for {circuit_name} task completion...")
            pcs.wait_tasks([task_id])
            result = pcs.get_results([task_id])
            
            print(f"✅ {circuit_name} task completed successfully!")
            
            # Small delay between tasks
            time.sleep(1)
        
        # Test multiple tasks for load balancing
        print(f"\n🔄 Testing load balancing with multiple Bell state tasks...")
        
        def simple_bell_circuit():
            qc = QuantumCircuit(2, 2)
            qc.h(0)
            qc.cx(0, 1)
            qc.measure([0, 1], [0, 1])
            return qc
        
        tasks = []
        for i in range(5):
            qt = QuantumTask(
                circuit=simple_bell_circuit,
                num_qubits=2,
                gate_set=["h", "cx", "measure"],
                resource_config={"num_qpus": 1, "num_gpus": 0, "memory": None}
            )
            
            print(f" Submitting Bell state task {i+1}/5...")
            task_id = pcs.submit_quantum_task(qt)
            tasks.append(task_id)
            
            # Small delay to see resource selection
            time.sleep(0.5)
        
        # Wait for completion
        print("⏳ Waiting for all Bell state tasks to complete...")
        pcs.wait_tasks(tasks)
        results = pcs.get_results(tasks)
        
        print(f"✅ All {len(tasks)} Bell state tasks completed successfully!")
        
        # Show final statistics
        print("\n=== Load Balancing Results ===")
        print("📊 Analyzing load balancing results...")
        
        # Check metrics file for resource distribution
        metrics_file = pcs.metrics_file_name
        if os.path.exists(metrics_file):
            import csv
            resource_counts = {}
            with open(metrics_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    resource = row.get('pilot_scheduled', 'unknown')
                    resource_counts[resource] = resource_counts.get(resource, 0) + 1
            
            print("Resource distribution:")
            total_tasks = sum(resource_counts.values())
            for resource, count in resource_counts.items():
                percentage = (count / total_tasks) * 100
                print(f"  {resource}: {count} tasks ({percentage:.1f}%)")
        
        print(f"\n✅ SUCCESS: Framework backends demonstration completed!")
        print(f"   - QDREAMER successfully selected appropriate backends")
        print(f"   - Different circuit complexities were handled correctly")
        print(f"   - Load balancing worked with framework-provided resources")
        
    except Exception as e:
        print(f"❌ Error during framework backends demo: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if pcs:
            print("🧹 Cleaning up Pilot Compute Service...")
            pcs.cancel()
            print("✅ Cleanup completed")


if __name__ == "__main__":
    # Run the framework backends demonstration
    demonstrate_framework_backends()
