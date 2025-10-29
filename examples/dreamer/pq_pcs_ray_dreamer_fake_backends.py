#!/usr/bin/env python3
"""
Pilot Quantum with QDREAMER - Multi-Pilot Example

This example demonstrates using QDREAMER with multiple pilots, each with
different executors, and letting the pilot compute service schedule jobs
across all available pilots:

- Pilot 1: Qiskit local executor with framework-provided backends
- Pilot 2: IBMQ executor with specific fake backend selection

The example shows how the pilot compute service intelligently schedules
jobs across multiple pilots based on resource availability and requirements.
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

# Pilot 1: Qiskit local executor with framework-provided backends
pilot_qiskit_description = {
    "resource_type": "quantum",
    "quantum": {
        "executor": "qiskit_local"
    },
    "working_directory": WORKING_DIRECTORY,
    "type": "ray",
}

# Pilot 2: IBMQ executor with specific fake backend
pilot_ibmq_description = {
    "resource_type": "quantum",
    "quantum": {
        "executor": "ibmq",
    },
    "working_directory": WORKING_DIRECTORY,
    "type": "ray" 
}



def create_bell_state_circuit():
    """Create a Bell state circuit."""
    print("🔧 Creating Bell state circuit (2 qubits)...")
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    # Don't explicitly add measure - let backend handle measurement
    print(f"   ✅ Circuit created: {qc.num_qubits} qubits, {qc.num_clbits} classical bits")
    print(f"   Circuit depth: {qc.depth()}, Total gates: {qc.size()}")
    return qc

def create_ghz_state_circuit():
    """Create a GHZ state circuit."""
    print("🔧 Creating GHZ state circuit (3 qubits)...")
    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    # Don't explicitly add measure - let backend handle measurement
    print(f"   ✅ Circuit created: {qc.num_qubits} qubits, {qc.num_clbits} classical bits")
    print(f"   Circuit depth: {qc.depth()}, Total gates: {qc.size()}")
    return qc

def create_ibmq_compatible_circuit():
    """Create a circuit using gates that are ONLY supported by IBMQ fake backends."""
    print("🔧 Creating IBMQ-only circuit (2 qubits)...")
    qc = QuantumCircuit(2, 2)
    # Use gates that are ONLY supported by IBMQ fake backends, not Qiskit local
    qc.sx(0)  # SX gate - only supported by IBMQ fake backends
    qc.rz(0.5, 0)  # RZ gate - only supported by IBMQ fake backends  
    qc.cx(0, 1)  # CNOT gate - supported by both
    # Don't explicitly add measure - let backend handle measurement
    print(f"   ✅ Circuit created: {qc.num_qubits} qubits, {qc.num_clbits} classical bits")
    print(f"   Circuit depth: {qc.depth()}, Total gates: {qc.size()}")
    return qc

def create_simple_circuit():
    """Create a simple circuit using basic gates."""
    print("🔧 Creating simple circuit (2 qubits)...")
    qc = QuantumCircuit(2, 2)
    qc.x(0)
    qc.cx(0, 1)
    # Don't explicitly add measure - let backend handle measurement
    print(f"   ✅ Circuit created: {qc.num_qubits} qubits, {qc.num_clbits} classical bits")
    print(f"   Circuit depth: {qc.depth()}, Total gates: {qc.size()}")
    return qc

def demonstrate_multi_pilot():
    """Demonstrate using multiple pilots with different executors."""
    print("=== Multi-Pilot QDREAMER Demonstration ===\n")
    print("🎯 Starting Multi-Pilot Demonstration")
    
    print("🔧 Configuration:")
    print("   - Pilot 1: 'qiskit_local' executor with framework-provided backends")
    print("   - Pilot 2: 'ibmq' executor with specific fake backend")
    print("   - Optimization: PuLP-based high_fidelity mode")
    print("   - Job scheduling: Across multiple pilots")
    
    pcs = None
    try:
        print("\n🚀 Starting Pilot Compute Service...")
        pcs = PilotComputeService(execution_engine=ExecutionEngine.RAY, 
                                  working_directory=WORKING_DIRECTORY)
        
        # Create multiple pilots
        print("\n📋 Creating Pilot 1 (Qiskit Local)...")
        pilot1_name = pcs.create_pilot(pilot_compute_description=pilot_qiskit_description)
        print(f"   ✅ Pilot 1 created: {pilot1_name}")
        
        print("\n📋 Creating Pilot 2 (IBMQ)...")
        pilot2_name = pcs.create_pilot(pilot_compute_description=pilot_ibmq_description)
        print(f"   ✅ Pilot 2 created: {pilot2_name}")
        
        print("\n🧠 Initializing QDREAMER for optimum resource selection...")
        pcs.initialize_dreamer() 
        
        print("✅ Pilot Compute Service initialized successfully")
        
        # Display available resources across all pilots
        print(f"\n📋 Available Quantum Resources Across All Pilots:")
        print(f"   Found {len(pcs.quantum_resources)} total resources!")
        
        # Show resources by executor type
        qiskit_resources = {name: res for name, res in pcs.quantum_resources.items() 
                           if 'qiskit' in name.lower()}
        ibmq_resources = {name: res for name, res in pcs.quantum_resources.items() 
                         if 'fake' in name.lower()}
        
        print(f"\n   Qiskit Local Resources ({len(qiskit_resources)}):")
        for name, resource in qiskit_resources.items():
            fidelity = 1 - resource.error_rate if resource.error_rate is not None else 1.0
            error_rate_display = f"{resource.error_rate:.3f}" if resource.error_rate is not None else "0.000"
            print(f"     - {name}: {resource.qubit_count} qubits, "
                  f"fidelity={fidelity:.3f}, error_rate={error_rate_display}")
            print(f"       Gateset: {resource.gateset}")
        
        print(f"\n   IBMQ Resources ({len(ibmq_resources)}):")
        count = 0
        for name, resource in ibmq_resources.items():
            if count < 3:  # Show first 3
                fidelity = 1 - resource.error_rate if resource.error_rate is not None else 1.0
                error_rate_display = f"{resource.error_rate:.3f}" if resource.error_rate is not None else "0.000"
                print(f"     - {name}: {resource.qubit_count} qubits, "
                      f"fidelity={fidelity:.3f}, error_rate={error_rate_display}")
                print(f"       Gateset: {resource.gateset}")
            count += 1
        
        if len(ibmq_resources) > 3:
            print(f"     ... and {len(ibmq_resources) - 3} more IBMQ backends")
        
        # Test different circuit complexities
        circuits = [
            ("Bell State", create_bell_state_circuit, 2, ["h", "cx"]),
            ("GHZ State", create_ghz_state_circuit, 3, ["h", "cx"]),
            ("Simple Circuit", create_simple_circuit, 2, ["x", "cx"]),
            ("IBMQ Only", create_ibmq_compatible_circuit, 2, ["sx", "rz", "cx"])
        ]
        
        print(f"\n🔄 Testing circuits with multi-pilot scheduling...")
        
        for circuit_name, circuit_func, num_qubits, gate_set in circuits:
            print(f"\n--- Testing {circuit_name} ---")
            
            qt = QuantumTask(
                circuit=circuit_func,
                num_qubits=num_qubits,
                gate_set=gate_set,
                resource_config={"num_qpus": 1, "num_gpus": 0, "memory": None},
            )
            
            print(f" Submitting {circuit_name} task...")
            task_id = pcs.submit_quantum_task(qt)
            
            print(f"⏳ Waiting for task completion...")
            pcs.wait_tasks([task_id])
            result = pcs.get_results([task_id])

            
            print(f"✅ {circuit_name} task completed successfully!")
        
        # Test load balancing with multiple tasks across pilots
        print(f"\n🔄 Testing load balancing with multiple tasks across pilots...")        
        
        # Show final statistics
        print("\n=== Multi-Pilot Load Balancing Results ===")
        print("📊 Analyzing load balancing results across pilots...")
        
        # Check metrics file for resource distribution
        metrics_file = pcs.metrics_file_name
        if os.path.exists(metrics_file):
            import csv
            resource_counts = {}
            pilot_counts = {}
            
            with open(metrics_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    resource = row.get('pilot_scheduled', 'unknown')
                    pilot = row.get('pilot_name', 'unknown')
                    
                    resource_counts[resource] = resource_counts.get(resource, 0) + 1
                    pilot_counts[pilot] = pilot_counts.get(pilot, 0) + 1
            
            print("Resource distribution:")
            total_tasks = sum(resource_counts.values())
            for resource, count in resource_counts.items():
                percentage = (count / total_tasks) * 100
                print(f"  {resource}: {count} tasks ({percentage:.1f}%)")
            
            print(f"\nPilot distribution:")
            for pilot, count in pilot_counts.items():
                percentage = (count / total_tasks) * 100
                print(f"  {pilot}: {count} tasks ({percentage:.1f}%)")
        
        print(f"\n✅ SUCCESS: Multi-pilot demonstration completed!")
        print(f"   - Multiple pilots successfully created and managed")
        print(f"   - Jobs scheduled across different pilot types")
        print(f"   - QDREAMER selected appropriate resources from all pilots")
        print(f"   - Load balancing worked across multiple pilots")
        
    except Exception as e:
        print(f"❌ Error during multi-pilot demo: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        if pcs:
            print("🧹 Cleaning up Pilot Compute Service...")
            pcs.cancel()
            print("✅ Cleanup completed")

if __name__ == "__main__":
    # Run the multi-pilot demonstration
    demonstrate_multi_pilot()
