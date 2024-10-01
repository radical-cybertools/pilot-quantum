import os
import time
from tracemalloc import start
import numpy as np
from zmq import device
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from pilot.pilot_compute_service import ExecutionEngine, PilotComputeService
from time import sleep
from qiskit_addon_cutting.automated_cut_finding import (
    find_cuts,
    OptimizationParameters,
    DeviceConstraints,
)

from qiskit_addon_cutting import partition_problem
from qiskit_addon_cutting import cut_wires, expand_observables

from qiskit_addon_cutting import generate_cutting_experiments
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2, Batch
from qiskit_addon_cutting import reconstruct_expectation_values
from qiskit_aer.primitives import EstimatorV2
from qiskit.circuit.library import EfficientSU2





RESOURCE_URL_HPC = "slurm://localhost"
WORKING_DIRECTORY = os.path.join(os.environ["HOME"], "work")

pilot_compute_description_ray = {
    "resource": RESOURCE_URL_HPC,
    "working_directory": WORKING_DIRECTORY,
    "type": "ray",
    "number_of_nodes": 2,
    "cores_per_node": 64,
    "gpus_per_node": 4,
    "project": "m4408",
    "queue": "premium",
    "walltime": 60,
    "scheduler_script_commands": ["#SBATCH --constraint=gpu", 
                                  "#SBATCH --gpus-per-task=1",
                                  "#SBATCH --ntasks-per-node=4",
                                  "#SBATCH --gpu-bind=none"],    
}

def start_pilot(pilot_compute_description_ray):
    pcs = PilotComputeService(execution_engine=ExecutionEngine.RAY, working_directory=WORKING_DIRECTORY)
    pcs.create_pilot(pilot_compute_description=pilot_compute_description_ray)
    return pcs


def pre_processing(scale=1):    
    base_qubuits = 8
    # circuit = random_circuit(base_qubuits *scale, 6, max_operands=2, seed=1242)
    # observable = SparsePauliOp(["ZIIIIIII"*scale, "IIIZIIII"*scale, "IIIIIIIZ"*scale])
    
    circuit = EfficientSU2(base_qubuits*scale, entanglement="linear", reps=2).decompose()
    circuit.assign_parameters([0.4] * len(circuit.parameters), inplace=True)
    observable = SparsePauliOp(["ZIIIIIII"*scale, "IIIZIIII"*scale, "IIIIIIIZ"*scale])    

    # Specify settings for the cut-finding optimizer
    optimization_settings = OptimizationParameters(seed=111)

    # Specify the size of the QPUs available
    device_constraints = DeviceConstraints(qubits_per_subcircuit=2)

    cut_circuit, metadata = find_cuts(circuit, optimization_settings, device_constraints)
    print(
        f'Found solution using {len(metadata["cuts"])} cuts with a sampling '
        f'overhead of {metadata["sampling_overhead"]}.\n'
        f'Lowest cost solution found: {metadata["minimum_reached"]}.'
    )
    for cut in metadata["cuts"]:
        print(f"{cut[0]} at circuit instruction index {cut[1]}")


    qc_w_ancilla = cut_wires(cut_circuit)
    observables_expanded = expand_observables(observable.paulis, circuit, qc_w_ancilla)

    partitioned_problem = partition_problem(
        circuit=qc_w_ancilla, observables=observables_expanded
    )
    subcircuits = partitioned_problem.subcircuits
    subobservables = partitioned_problem.subobservables
    print(
        f"Sampling overhead: {np.prod([basis.overhead for basis in partitioned_problem.bases])}"
    )

    subexperiments, coefficients = generate_cutting_experiments(
        circuits=subcircuits, observables=subobservables, num_samples=1_000
    )
    print(
        f"{len(subexperiments[0]) + len(subexperiments[1])} total subexperiments to run on backend."
    )
    return subexperiments, coefficients, subobservables, observable, circuit

def execute_sampler(sampler, label, subsystem_subexpts, shots):
    submit_start = time.time()
    job = sampler.run(subsystem_subexpts, shots=shots)
    submit_end = time.time()
    result_start = time.time()
    result = job.result()    
    result_end = time.time()
    print(f"Job {label} completed with job id {job.job_id()}, submit_time: {submit_end-submit_start} and execution_time: {result_end - result_start}, type: {type(result)}")
    return (label, result)

if __name__ == "__main__":
    pcs = None
    num_nodes = [4]
    for scale in [4]:
        start_time = time.time()
        try:
            # Start Pilot
            pilot_compute_description_ray["number_of_nodes"] = num_nodes[0]
            pcs = start_pilot(pilot_compute_description_ray)
            
            subexperiments, coefficients, subobservables, observable, circuit = pre_processing(scale)
            
            backend = AerSimulator(device="GPU")
            print("*********************************** transpiling circuits ***********************************")
            # Transpile the subexperiments to ISA circuits
            pass_manager = generate_preset_pass_manager(optimization_level=1, backend=backend)
            isa_subexperiments = {}
            for label, partition_subexpts in subexperiments.items():
                isa_subexperiments[label] = pass_manager.run(partition_subexpts)
            print("*********************************** transpiling done ***********************************")
            
            tasks=[]
            i=0
            sub_circuit_execution_time = time.time()
            with Batch(backend=backend) as batch:
                sampler = SamplerV2(mode=batch)
                print(f"*********************************** len of subexperiments {len(isa_subexperiments)}*************************")
                for label, subsystem_subexpts in isa_subexperiments.items():
                    task_future = pcs.submit_task(execute_sampler, sampler, label, subsystem_subexpts, shots=2**12, resources={'num_cpus': 1, 'num_gpus': 1, 'memory': None})
                    tasks.append(task_future)
                    i=i+1

            results_tuple=pcs.get_results(tasks)
            # print(results_tuple)
            sub_circuit_execution_end_time = time.time()
            print("Execution time for subcircuits: ", sub_circuit_execution_end_time-sub_circuit_execution_time)
            
            results = {}
            
            for result in results_tuple:
                results[result[0]] = result[1]
            
            reconstruction_start_time = time.time()
            # Get expectation values for each observable term
            reconstructed_expvals = reconstruct_expectation_values(
                results,
                coefficients,
                subobservables,
            )
            reconstruction_end_time = time.time()
            print("Execution time for reconstruction: ", reconstruction_end_time-reconstruction_start_time)
                        
            final_expval = np.dot(reconstructed_expvals, observable.coeffs)
            print(f"Reconstructed expectation value: {np.real(np.round(final_expval, 8))}")
            
            full_circuit_estimator_time = time.time()
            estimator = EstimatorV2()
            exact_expval = estimator.run([(circuit, observable)]).result()[0].data.evs
            full_circuit_estimator_end_time = time.time()
            print("Execution time for full Circuit: ", full_circuit_estimator_end_time-full_circuit_estimator_time)
            
            print(f"Reconstructed expectation value: {np.real(np.round(final_expval, 8))}")
            print(f"Exact expectation value: {np.round(exact_expval, 8)}")
            print(f"Error in estimation: {np.real(np.round(final_expval-exact_expval, 8))}")
            print(
                f"Relative error in estimation: {np.real(np.round((final_expval-exact_expval) / exact_expval, 8))}"
            )

            end_time = time.time()
            print(f"Execution time for {num_nodes[0]} nodes: {end_time-start_time}")                                           
        except Exception as e:
            print(e)
        finally:
            if pcs:
                pcs.cancel()
