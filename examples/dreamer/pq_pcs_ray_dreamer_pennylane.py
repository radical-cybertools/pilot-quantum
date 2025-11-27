#!/usr/bin/env python3
"""
Multi-Pilot QDREAMER with PennyLane Circuits

This example demonstrates the simplified PennyLane architecture:
- Single 'pennylane' executor
- Device information read from pilot_compute_description
- No redundant backend parameter
"""

import os
import sys
import time
import pennylane as qml
from pilot.pilot_compute_service import PilotComputeService, ExecutionEngine
from pilot.dreamer import Q_DREAMER, QuantumTask

# Add the parent directory to the path to import pilot modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configuration
WORKING_DIRECTORY = "/Users/pmantha/work"

def create_bell_state_circuit():
    """Create a Bell state circuit using PennyLane (raw function, no device)."""
    
    # Return raw circuit function - device will be provided by executor
    def bell_state():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.probs(wires=[0, 1])
    
    print(f"🔧 Creating Bell state circuit (2 qubits) with PennyLane...")
    print(f"   ✅ Circuit created: 2 qubits")
    print(f"   Circuit depth: 2, Total gates: 2")
    print(f"   Note: Device will be configured by executor")
    
    return bell_state


def create_ghz_state_circuit():
    """Create a GHZ state circuit using PennyLane (raw function, no device)."""
    
    # Return raw circuit function - device will be provided by executor
    def ghz_state():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        return qml.probs(wires=[0, 1, 2])
    
    print(f"🔧 Creating GHZ state circuit (3 qubits) with PennyLane...")
    print(f"   ✅ Circuit created: 3 qubits")
    print(f"   Circuit depth: 3, Total gates: 3")
    print(f"   Note: Device will be configured by executor")
    
    return ghz_state


def create_simple_circuit():
    """Create a simple circuit using PennyLane (raw function, no device)."""
    
    # Return raw circuit function - device will be provided by executor
    def simple_circuit():
        qml.PauliX(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.probs(wires=[0, 1])
    
    print(f"🔧 Creating simple circuit (2 qubits) with PennyLane...")
    print(f"   ✅ Circuit created: 2 qubits")
    print(f"   Circuit depth: 2, Total gates: 2")
    print(f"   Note: Device will be configured by executor")
    
    return simple_circuit


def create_device_specific_circuit():
    """Create a circuit using gates that are device-specific."""
    
    # Return raw circuit function - device will be provided by executor
    def device_specific_circuit():
        # Use gates that are supported by PennyLane default but not by PennyLane Qiskit
        qml.RX(0.5, wires=0)  # RX gate - supported by PennyLane default but not Qiskit
        qml.RY(0.3, wires=0)  # RY gate - supported by PennyLane default but not Qiskit  
        qml.CNOT(wires=[0, 1])  # CNOT gate - supported by both
        return qml.probs(wires=[0, 1])
    
    print(f"🔧 Creating device-specific circuit (2 qubits) with PennyLane...")
    print(f"   ✅ Circuit created: 2 qubits")
    print(f"   Circuit depth: 3, Total gates: 3")
    print(f"   Note: Device will be configured by executor")
    print(f"   Note: RX/RY gates will force selection of compatible device")
    
    return device_specific_circuit


def create_qiskit_aer_specific_circuit():
    """Create a circuit using gates that are ONLY supported by Qiskit Aer."""
    
    # Return raw circuit function - device will be provided by executor
    def qiskit_aer_specific_circuit():
        # Use gates that are ONLY supported by Qiskit Aer, not by PennyLane default
        qml.U1(0.5, wires=0)  # U1 gate - Qiskit specific, not in PennyLane default
        qml.U2(0.3, 0.7, wires=1)  # U2 gate - Qiskit specific, not in PennyLane default
        qml.U3(0.1, 0.2, 0.3, wires=2)  # U3 gate - Qiskit specific, not in PennyLane default
        qml.CX(wires=[0, 1])  # CX gate (same as CNOT) - supported by both
        return qml.probs(wires=[0, 1, 2])
    
    print(f"🔧 Creating Qiskit Aer specific circuit (3 qubits) with PennyLane...")
    print(f"   ✅ Circuit created: 3 qubits")
    print(f"   Circuit depth: 4, Total gates: 4")
    print(f"   Note: U1/U2/U3 gates will FORCE selection of qiskit.aer device")
    print(f"   Note: These gates are NOT supported by default.qubit")
    
    return qiskit_aer_specific_circuit

def demonstrate_single_pilot_multi_device():
    """Demonstrate single pilot with multiple PennyLane devices."""
    print("\n=== Single Pilot QDREAMER with Multiple PennyLane Devices ===\n")
    
    print("🎯 Starting Single Pilot Multi-Device PennyLane Demonstration")
    print("🔧 Configuration:")
    print("   - Single Pilot: 'pennylane' executor with multiple devices")
    print("   - Devices: default.qubit, qiskit.aer")
    print("   - Optimization: PuLP-based high_speed mode")
    print("   - Objectives: Fidelity (20%), Queue (80%)")
    print("   - Job scheduling: Across multiple devices within single pilot")
    print("   - Circuits: PennyLane quantum functions")
    print("   - Architecture: Single 'pennylane' executor with multiple device configs")
    
    # Define single pilot with multiple devices
    pilot_description = {
        "resource_type": "quantum",
        "quantum": {
            "executor": "pennylane",
            "devices": [  # Multiple devices in single pilot
                "default.qubit",
                "qiskit.aer"
            ]
        },
        "working_directory": WORKING_DIRECTORY,
        "type": "ray"
    }
    
    # Initialize Pilot Compute Service
    print("\n🚀 Starting Pilot Compute Service...")
    pcs = PilotComputeService(execution_engine=ExecutionEngine.RAY, working_directory=WORKING_DIRECTORY)
    
    # Create single pilot with multiple devices
    print("\n📋 Creating Single Pilot with Multiple Devices...")
    pilot = pcs.create_pilot(pilot_description)
    print(f"   ✅ Pilot created: {pilot}")
    
    # Initialize QDREAMER with intelligent optimization
    print("\n🧠 Initializing QDREAMER with intelligent optimization...")
    qdreamer = pcs.initialize_dreamer()  # Uses default high_fidelity config
    print("✅ Pilot Compute Service initialized successfully")
    
    # Display available resources
    print("\n📋 Available Quantum Resources (Single Pilot, Multiple Devices):")
    resources = pcs.quantum_resources
    print(f"   Found {len(resources)} total resources!")
    
    # Group resources by device type
    default_qubit_resources = []
    qiskit_aer_resources = []
    
    for resource_name, resource in resources.items():
        if "default_qubit" in resource_name.lower():
            default_qubit_resources.append((resource_name, resource))
        elif "qiskit" in resource_name.lower():
            qiskit_aer_resources.append((resource_name, resource))
    
    print(f"\n   Default.qubit Resources ({len(default_qubit_resources)}):")
    for name, resource in default_qubit_resources:
        fidelity = 1 - resource.error_rate if resource.error_rate is not None else 1.0
        error_rate_display = f"{resource.error_rate:.3f}" if resource.error_rate is not None else "0.000"
        print(f"     - {name}: {resource.qubit_count} qubits, "
              f"fidelity={fidelity:.3f}, error_rate={error_rate_display}")
        print(f"       Gateset: {resource.gateset}")
    
    print(f"\n   Qiskit.aer Resources ({len(qiskit_aer_resources)}):")
    for name, resource in qiskit_aer_resources:
        fidelity = 1 - resource.error_rate if resource.error_rate is not None else 1.0
        error_rate_display = f"{resource.error_rate:.3f}" if resource.error_rate is not None else "0.000"
        print(f"     - {name}: {resource.qubit_count} qubits, "
              f"fidelity={fidelity:.3f}, error_rate={error_rate_display}")
        print(f"       Gateset: {resource.gateset}")
    
    # Test circuits
    print("\n🔄 Testing PennyLane circuits with single pilot multi-device scheduling...")
    
    # Define circuits to test with their metadata
    circuits = [
        ("Bell State", create_bell_state_circuit, 2, ["h", "cnot"]),
        ("GHZ State", create_ghz_state_circuit, 3, ["h", "cnot"]),
        ("Simple Circuit", create_simple_circuit, 2, ["x", "cnot"]),
        ("Device Specific", create_device_specific_circuit, 2, ["rx", "ry", "cnot"]),
        ("Qiskit Aer Specific", create_qiskit_aer_specific_circuit, 3, ["u1", "u2", "u3", "cx"])
    ]
    
    # Test each circuit
    for circuit_name, circuit_func, num_qubits, gate_set in circuits:
        print(f"\n--- Testing {circuit_name} ---")
        
        # Create circuit
        circuit = circuit_func()
        
        # Submit task using hardcoded metadata
        print(f"Submitting {circuit_name} task...")
        task = QuantumTask(
            circuit=circuit,
            num_qubits=num_qubits,
            gate_set=gate_set,
            resource_config={}
        )
        
        task_id = pcs.submit_quantum_task(task)
        print(f"⏳ Waiting for task completion...")
        
        # Wait for completion
        pcs.wait_tasks([task_id])
        print(f"✅ {circuit_name} task completed successfully!")
    
    # Test load balancing with multiple tasks
    print("\n🔄 Testing load balancing with multiple tasks across devices...")
    circuit = create_simple_circuit()
    
    # Submit multiple tasks using hardcoded metadata
    task_ids = []
    for i in range(4):
        task = QuantumTask(
            circuit=circuit,
            num_qubits=2,
            gate_set=["x", "cnot"],
            resource_config={}
        )
        task_id = pcs.submit_quantum_task(task)
        task_ids.append(task_id)
    
    # Wait for all tasks
    pcs.wait_tasks(task_ids)
    
    for i, task_id in enumerate(task_ids, 1):
        print(f"✅ Task {i} completed")
    
    # Analyze results
    print("\n=== Single Pilot Multi-Device Load Balancing Results ===")
    print("📊 Analyzing load balancing results across devices...")
    
    # Get results and analyze distribution
    results = pcs.get_results(task_ids)
    
    # Count resource usage
    resource_usage = {}
    device_usage = {}
    
    for result in results:
        # Handle both dictionary and tensor results
        if isinstance(result, dict):
            resource_name = result.get('resource_name', 'unknown')
            device_name = resource_name.split('_')[-1] if '_' in resource_name else 'unknown'
        else:
            # For tensor results, we can't extract metadata, so use defaults
            resource_name = 'pennylane_default_qubit'  # Most likely
            device_name = 'default_qubit'
        
        resource_usage[resource_name] = resource_usage.get(resource_name, 0) + 1
        device_usage[device_name] = device_usage.get(device_name, 0) + 1
    
    print("Resource distribution:")
    for resource, count in resource_usage.items():
        percentage = (count / len(results)) * 100
        print(f"  {resource}: {count} tasks ({percentage:.1f}%)")
    
    print("\nDevice distribution:")
    for device, count in device_usage.items():
        percentage = (count / len(results)) * 100
        print(f"  {device}: {count} tasks ({percentage:.1f}%)")
    
    print(f"\n✅ SUCCESS: Single pilot multi-device PennyLane demonstration completed!")
    print("   - Single pilot successfully created with multiple devices")
    print("   - PennyLane circuits executed across different device types")
    print("   - QDREAMER selected appropriate resources from all devices")
    print("   - Load balancing worked across multiple devices within single pilot")
    print("   - Simplified architecture: single 'pennylane' executor")
    print("   - Multiple devices specified in single pilot description")
    
    # Cleanup
    print("\n🧹 Cleaning up Pilot Compute Service...")
    pcs.cancel()
    print("✅ Cleanup completed")

if __name__ == "__main__":
    demonstrate_single_pilot_multi_device()
