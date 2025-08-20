import math
import os
import time
import random
from typing import Dict, List

import pennylane as qml
import ray
from pilot.dreamer import Coupling, QuantumTask, TaskType
from pilot.pilot_compute_service import ExecutionEngine, PilotComputeService
from pilot.util.quantum_resource_generator import QuantumResourceGenerator, QuantumResource
from time import sleep
from qiskit import QuantumCircuit

RESOURCE_URL_HPC = "ssh://localhost"
WORKING_DIRECTORY = os.path.join(os.environ["HOME"], "work")

# Enhanced quantum description with multiple backend simulators
pilot_quantum_description_multi_backend = {
    "resource_type": "quantum",
    "quantum": {
        "executor": "qiskit_local",  # Base executor
        "custom_backends": {
            "qasm_simulator_high_fidelity": {
                "fidelity_score": 0.999,
                "error_rate": 0.001,
                "noise_level": 0.001,
                "qubit_count": 32,
                "queue_length": 5
            },
            "qasm_simulator_medium_fidelity": {
                "fidelity_score": 0.95,
                "error_rate": 0.05,
                "noise_level": 0.03,
                "qubit_count": 64,
                "queue_length": 2
            },
            "qasm_simulator_low_fidelity": {
                "fidelity_score": 0.85,
                "error_rate": 0.15,
                "noise_level": 0.10,
                "qubit_count": 128,
                "queue_length": 0
            }
        }
    },
    "working_directory": WORKING_DIRECTORY,
    "type": "ray",
    "dreamer_enabled": True,
}

# QDREAMER configuration is now simplified - just pass optimization_mode to initialize_dreamer()

def start_pilot():
    """Initialize Pilot Compute Service with multi-backend quantum resources."""
    print("🚀 Starting Pilot Compute Service initialization...")
    
    pcs = PilotComputeService(execution_engine=ExecutionEngine.RAY, 
                              working_directory=WORKING_DIRECTORY)
    
    print("📋 Creating quantum pilot with multiple backends...")
    pcs.create_pilot(pilot_compute_description=pilot_quantum_description_multi_backend)
    
    print("🧠 Initializing QDREAMER for intelligent resource selection...")
    pcs.initialize_dreamer({"optimization_mode": "high_fidelity"})
    
    print("✅ Pilot Compute Service initialized successfully")
    return pcs

def demonstrate_load_balancing():
    """Demonstrate QDREAMER's load balancing capabilities."""
    print("=== QDREAMER Load Balancing Demonstration ===\n")
    print("🎯 Starting QDREAMER Load Balancing Demonstration")
    
    pcs = None
    try:
        # Start Pilot with enhanced QDREAMER
        pcs = start_pilot()
        
        # Display available resources (created by framework)
        print(f"\n📋 Available quantum resources (created by framework):")
        for name, resource in pcs.quantum_resources.items():
            fidelity = 1 - resource.error_rate
            print(f"   - {name}: {resource.qubit_count} qubits, "
                  f"fidelity={fidelity:.3f}, error_rate={resource.error_rate:.3f}")
        
        # Create a simple circuit for testing
        def simple_circuit():
            qc = QuantumCircuit(2, 2)
            qc.h(0)
            qc.cx(0, 1)
            qc.measure([0, 1], [0, 1])
            return qc
        
        # Submit multiple tasks to see load balancing in action
        print("\nSubmitting 10 tasks to observe load balancing...")
        print("🔄 Submitting 10 tasks to observe load balancing behavior...")
        
        tasks = []
        for i in range(10):
            qt = QuantumTask(
                circuit=simple_circuit,
                num_qubits=2,
                gate_set=["h", "cx", "measure"],
                resource_config={"num_qpus": 1, "num_gpus": 0, "memory": None}
            )
            
            print(f" Submitting task {i+1}/10...")
            task_id = pcs.submit_quantum_task(qt)
            tasks.append(task_id)
            
            # Small delay to see resource selection
            time.sleep(0.5)
        
        # Wait for completion
        print("⏳ Waiting for all tasks to complete...")
        pcs.wait_tasks(tasks)
        results = pcs.get_results(tasks)
        
        print(f"✅ All {len(tasks)} tasks completed successfully!")
        
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
            for resource, count in resource_counts.items():
                percentage = (count / len(tasks)) * 100
                print(f"  {resource}: {count} tasks ({percentage:.1f}%)")
        
    except Exception as e:
        print(f"Error during load balancing demo: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if pcs:
            print("🧹 Cleaning up Pilot Compute Service...")
            pcs.cancel()
            print("✅ Cleanup completed")

if __name__ == "__main__":
    # Run the load balancing demonstration
    demonstrate_load_balancing()
